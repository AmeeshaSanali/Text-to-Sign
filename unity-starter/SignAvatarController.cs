using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// SignAvatarController controls sign language animation playback on a rigged humanoid avatar.
/// Implements Track A Phase A: Typed-word playback queue for continuous sequential animation.
/// Seamlessly blends between retargeted 3D pose animations (PoseJsonLoader/SignPlayer)
/// and embedded clip animations (Salute, Shooting_Arrow fallback).
/// </summary>
public class SignAvatarController : MonoBehaviour
{
    [System.Serializable]
    public class QueuedSign
    {
        public string gloss;
        public string clipName;
        public bool hasPoseAnimation;
        public string animationUrl;
        public string signStatus; // "approved", "clip_fallback", "no_sign"
        public bool isUnknown;

        public QueuedSign(string gloss, string clipName = null, bool hasPose = false, string url = null, string status = "clip_fallback")
        {
            this.gloss = gloss;
            this.clipName = string.IsNullOrEmpty(clipName) ? gloss : clipName;
            this.hasPoseAnimation = hasPose;
            this.animationUrl = url;
            this.signStatus = status;
            this.isUnknown = (status == "no_sign");
        }
    }

    [Header("Components")]
    [SerializeField] private Animator animator;
    [SerializeField] private SignPlayer signPlayer;
    [SerializeField] private PoseJsonLoader poseLoader;

    [Header("Backend API Configuration")]
    [SerializeField] private string backendUrl = "http://127.0.0.1:8000/api/v1/translate";
    [SerializeField] private float transitionDuration = 0.2f;
    [SerializeField] private float pauseBetweenSigns = 0.25f;

    [Header("Quick Start Test (Day 3 Hardcoded)")]
    [SerializeField] private string[] testGlossSequence = new string[] { "Salute", "Shooting_Arrow" };

    [Header("Live Status")]
    public string currentStatus = "Ready";

    // Typed-word playback queue
    private Queue<QueuedSign> signQueue = new Queue<QueuedSign>();
    private Coroutine queueWorkerRoutine;
    private string lastProcessedSentence = "";
    private int lastQueuedCount = 0;
    private bool isTranslating = false;

    // Public state properties for UI inspection
    public QueuedSign CurrentPlayingSign { get; private set; }
    public List<QueuedSign> QueuedSigns => new List<QueuedSign>(signQueue);
    public int QueueCount => signQueue.Count;
    public bool IsQueueActive => queueWorkerRoutine != null;

    private Vector3 initialPosition;
    private Quaternion initialRotation;

    private void Awake()
    {
        if (animator == null) animator = GetComponent<Animator>();
        if (signPlayer == null) signPlayer = GetComponent<SignPlayer>() ?? GetComponentInChildren<SignPlayer>() ?? gameObject.AddComponent<SignPlayer>();
        if (poseLoader == null) poseLoader = GetComponent<PoseJsonLoader>() ?? GetComponentInChildren<PoseJsonLoader>() ?? gameObject.AddComponent<PoseJsonLoader>();

        // Lock root motion so animations do not displace character off-camera
        if (animator != null)
        {
            animator.applyRootMotion = false;
        }

        initialPosition = transform.position;
        initialRotation = transform.rotation;
    }

    private void LateUpdate()
    {
        // Keep character rooted in place facing camera
        if (queueWorkerRoutine != null && animator != null && !animator.applyRootMotion)
        {
            transform.position = initialPosition;
            transform.rotation = initialRotation;
        }
    }

    #region Typed-Word Queue Management

    /// <summary>
    /// Called by UI when a finished word is detected (e.g. user typed a space, punctuation, or Enter).
    /// Sends full sentence to /translate so grammar reordering works, but only enqueues newly added signs.
    /// If earlier text was modified or deleted, clears queue and restarts.
    /// </summary>
    public void OnTypedWordFinished(string fullSentenceText)
    {
        if (string.IsNullOrWhiteSpace(fullSentenceText))
        {
            ClearQueue();
            return;
        }

        string trimmed = fullSentenceText.Trim();

        // Check if this is an extension of the existing sentence or an edit/replacement
        bool isExtension = !string.IsNullOrEmpty(lastProcessedSentence) && trimmed.StartsWith(lastProcessedSentence);

        if (!isExtension && trimmed != lastProcessedSentence)
        {
            // Earlier text was edited: clear the upcoming queue to stay in sync
            Debug.Log("[SignAvatarController] Sentence was edited. Clearing pending queue.");
            ClearPendingQueue();
            lastQueuedCount = 0;
        }

        StartCoroutine(SendTranslationRequest(trimmed, isExtension));
    }

    /// <summary>
    /// Enqueues a list of gloss names into the playback queue.
    /// Starts the queue worker coroutine if idle.
    /// </summary>
    public void EnqueueSigns(IEnumerable<string> glosses)
    {
        foreach (string g in glosses)
        {
            EnqueueSign(g);
        }
    }

    /// <summary>
    /// Enqueues a single sign into the playback queue.
    /// </summary>
    public void EnqueueSign(string gloss)
    {
        if (string.IsNullOrWhiteSpace(gloss)) return;

        string cleanGloss = gloss.Trim().ToUpper();
        string matchedClip = FindMatchingState(cleanGloss);
        bool hasClip = matchedClip != null;

        // Check if we have a retargeted dataset pose animation (in StreamingAssets)
        string local1 = System.IO.Path.Combine(Application.streamingAssetsPath, $"{cleanGloss}.json");
        string local2 = System.IO.Path.Combine(Application.streamingAssetsPath, "animations", "slsl", $"{cleanGloss}.json");
        bool hasPose = System.IO.File.Exists(local1) || System.IO.File.Exists(local2);

        // Create queued sign item
        QueuedSign item = new QueuedSign(
            gloss: cleanGloss,
            clipName: matchedClip,
            hasPose: hasPose,
            url: $"/animations/slsl/{cleanGloss}.json",
            status: hasPose ? "approved" : (hasClip ? "clip_fallback" : "unknown")
        );

        signQueue.Enqueue(item);
        Debug.Log($"[SignAvatarController] Enqueued sign: '{cleanGloss}' (hasPose: {hasPose}, clip: {matchedClip}, queue length: {signQueue.Count})");

        EnsureWorkerRunning();
    }

    /// <summary>
    /// Clears both upcoming queue and currently playing animation, returning to Idle.
    /// </summary>
    public void ClearQueue()
    {
        signQueue.Clear();
        lastQueuedCount = 0;
        lastProcessedSentence = "";
        CurrentPlayingSign = null;

        if (queueWorkerRoutine != null)
        {
            StopCoroutine(queueWorkerRoutine);
            queueWorkerRoutine = null;
        }

        ReturnToIdle();
        currentStatus = "Queue Cleared (Ready)";
    }

    /// <summary>
    /// Clears only pending queued signs, allowing current sign to finish playing.
    /// </summary>
    public void ClearPendingQueue()
    {
        signQueue.Clear();
        lastQueuedCount = 0;
    }

    private void EnsureWorkerRunning()
    {
        if (queueWorkerRoutine == null)
        {
            queueWorkerRoutine = StartCoroutine(QueueWorkerCoroutine());
        }
    }

    /// <summary>
    /// Continuous worker that consumes signs from the queue one-by-one.
    /// New words can arrive and be enqueued while an earlier sign is playing.
    /// </summary>
    private IEnumerator QueueWorkerCoroutine()
    {
        currentStatus = "Processing queue...";

        while (signQueue.Count > 0)
        {
            QueuedSign sign = signQueue.Dequeue();
            CurrentPlayingSign = sign;

            // Handle unrecognized/unapproved signs without throwing
            if (sign.isUnknown)
            {
                currentStatus = $"Skipping unknown sign: {sign.gloss}";
                Debug.LogWarning($"[SignAvatarController] Skipping unknown sign: '{sign.gloss}'");
                yield return new WaitForSeconds(0.2f);
                continue;
            }

            currentStatus = $"Playing: {sign.gloss}";
            Debug.Log($"[SignAvatarController] Playing '{sign.gloss}' from queue. Remaining: {signQueue.Count}");

            bool playedSuccessfully = false;

            // 1. Try Pose JSON animation via SignPlayer (Phase 4/5 3D retargeted data)
            if (signPlayer != null)
            {
                yield return signPlayer.PlaySingleSign(sign.gloss, success => playedSuccessfully = success);
            }

            // 2. If no Pose JSON or SignPlayer not present, fall back to Animator clip mode
            if (!playedSuccessfully)
            {
                yield return StartCoroutine(PlayClipSignAndWait(sign.clipName ?? sign.gloss));
            }

            // Natural pause between consecutive signs
            if (pauseBetweenSigns > 0 && signQueue.Count > 0)
            {
                yield return new WaitForSeconds(pauseBetweenSigns);
            }
        }

        CurrentPlayingSign = null;
        ReturnToIdle();
        currentStatus = "Ready";
        queueWorkerRoutine = null;
    }

    #endregion

    #region Clip Fallback Playback

    /// <summary>
    /// Searches for a matching animation state in Animator layer 0.
    /// </summary>
    public string FindMatchingState(string name)
    {
        if (animator == null || string.IsNullOrWhiteSpace(name)) return null;

        string clean = name.TrimStart('#').Trim();

        List<string> candidates = new List<string>
        {
            clean,
            clean.ToUpper(),
            clean.ToLower(),
            clean.Length > 1 ? char.ToUpper(clean[0]) + clean.Substring(1).ToLower() : clean.ToUpper(),
            clean.Replace("_", " "),
            clean.Replace(" ", "_"),
            clean.ToUpper().Replace(" ", "_"),
            clean.ToLower().Replace(" ", "_"),
        };

        // Semantic aliases for Mixamo FBX clips
        if (clean.Equals("SHOOT", StringComparison.OrdinalIgnoreCase) || 
            clean.Equals("ARROW", StringComparison.OrdinalIgnoreCase) ||
            clean.Equals("SHOOTING", StringComparison.OrdinalIgnoreCase) ||
            clean.Equals("SHOOTING_ARROW", StringComparison.OrdinalIgnoreCase) ||
            clean.Equals("SHOOTING ARROW", StringComparison.OrdinalIgnoreCase))
        {
            candidates.Add("Shooting_Arrow");
            candidates.Add("Shooting Arrow");
            candidates.Add("shooting_arrow");
            candidates.Add("SHOOTING_ARROW");
        }

        if (clean.Equals("SALUTE", StringComparison.OrdinalIgnoreCase))
        {
            candidates.Add("Salute");
            candidates.Add("salute");
            candidates.Add("SALUTE");
        }

        foreach (string candidate in candidates)
        {
            int hash = Animator.StringToHash(candidate);
            if (animator.HasState(0, hash))
            {
                return candidate;
            }
        }

        return null;
    }

    public bool PlaySign(string glossName)
    {
        if (animator == null)
        {
            Debug.LogError("[SignAvatarController] Animator component missing!");
            return false;
        }

        string matchedState = FindMatchingState(glossName);
        if (matchedState != null)
        {
            currentStatus = $"Playing: {matchedState}";
            animator.CrossFadeInFixedTime(matchedState, transitionDuration);
            return true;
        }

        Debug.LogWarning($"[SignAvatarController] Animation state for '{glossName}' not found.");
        return false;
    }

    private IEnumerator PlayClipSignAndWait(string glossName)
    {
        string matchedState = FindMatchingState(glossName);
        if (matchedState == null)
        {
            Debug.LogWarning($"[SignAvatarController] No matching clip state for '{glossName}'. Skipping.");
            yield break;
        }

        currentStatus = $"Playing (Clip): {matchedState}";
        animator.CrossFadeInFixedTime(matchedState, transitionDuration);
        yield return new WaitForSeconds(transitionDuration);

        float estimatedDuration = 2.0f;
        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        if (stateInfo.length > 0)
        {
            estimatedDuration = stateInfo.length;
        }

        float elapsed = 0f;
        float timeout = estimatedDuration + 1.0f;

        while (elapsed < timeout)
        {
            elapsed += Time.deltaTime;
            AnimatorStateInfo currentInfo = animator.GetCurrentAnimatorStateInfo(0);
            if (currentInfo.IsName(matchedState) && currentInfo.normalizedTime >= 0.95f)
            {
                break;
            }
            yield return null;
        }
    }

    private void ReturnToIdle()
    {
        if (animator == null) return;
        string idleState = FindMatchingState("Idle");
        if (idleState != null)
        {
            animator.CrossFadeInFixedTime(idleState, transitionDuration);
        }
        transform.position = initialPosition;
        transform.rotation = initialRotation;
    }

    #endregion

    #region Legacy & Test Helpers

    [ContextMenu("Play Hardcoded Test Sequence")]
    public void PlayTestSequence()
    {
        EnqueueSigns(testGlossSequence);
    }

    public void PlaySequence(IEnumerable<string> glosses)
    {
        ClearPendingQueue();
        EnqueueSigns(glosses);
    }

    public void TranslateAndPlay(string sentence)
    {
        ClearQueue();
        OnTypedWordFinished(sentence);
    }

    #endregion

    #region Backend HTTP Translation

    private IEnumerator SendTranslationRequest(string sentence, bool isExtension)
    {
        if (isTranslating) yield break;
        isTranslating = true;

        string jsonPayload = $"{{\"text\": \"{EscapeJson(sentence)}\", \"enable_reordering\": true, \"fingerspell_unknown\": true}}";

        using (UnityWebRequest request = new UnityWebRequest(backendUrl, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonPayload);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            isTranslating = false;

            if (request.result != UnityWebRequest.Result.Success)
            {
                currentStatus = "Offline Mode (Local Dataset Signs)";
                Debug.LogWarning($"[SignAvatarController] Backend API unavailable ({request.error}). Using local dataset sign queue.");

                // Tokenize input sentence by space/punctuation and enqueue directly from local dataset frames
                string[] words = sentence.Split(new char[] { ' ', ',', '.', '!', '?', ';', ':', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                int startIndex = isExtension ? Mathf.Clamp(lastQueuedCount, 0, words.Length) : 0;
                for (int i = startIndex; i < words.Length; i++)
                {
                    string word = words[i].Trim().ToUpper();
                    if (!string.IsNullOrEmpty(word))
                    {
                        EnqueueSign(word);
                    }
                }
                lastQueuedCount = words.Length;
                lastProcessedSentence = sentence;
                EnsureWorkerRunning();
                yield break;
            }

            string jsonResponse = request.downloadHandler.text;
            BackendResponse parsed = JsonUtility.FromJson<BackendResponse>(jsonResponse);

            if (parsed == null || parsed.animation_sequence == null || parsed.animation_sequence.Length == 0)
            {
                if (parsed != null && parsed.glosses != null && parsed.glosses.Length > 0)
                {
                    // Fallback if animation_sequence field missing
                    int startIndex = isExtension ? lastQueuedCount : 0;
                    for (int i = startIndex; i < parsed.glosses.Length; i++)
                    {
                        EnqueueSign(parsed.glosses[i]);
                    }
                    lastQueuedCount = parsed.glosses.Length;
                    lastProcessedSentence = sentence;
                }
                yield break;
            }

            // Enqueue new cues from animation_sequence
            int startIdx = isExtension ? Mathf.Clamp(lastQueuedCount, 0, parsed.animation_sequence.Length) : 0;
            for (int i = startIdx; i < parsed.animation_sequence.Length; i++)
            {
                var cue = parsed.animation_sequence[i];
                QueuedSign item = new QueuedSign(
                    gloss: cue.gloss,
                    clipName: cue.animation_clip,
                    hasPose: cue.has_pose_animation,
                    url: cue.animation_url,
                    status: cue.sign_status ?? "clip_fallback"
                );
                signQueue.Enqueue(item);
                Debug.Log($"[SignAvatarController] Enqueued sign from API: {cue.gloss} (Status: {cue.sign_status})");
            }

            lastQueuedCount = parsed.animation_sequence.Length;
            lastProcessedSentence = sentence;
            EnsureWorkerRunning();
        }
    }

    private string EscapeJson(string str)
    {
        return str.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r");
    }

    [Serializable]
    public class AnimationCueData
    {
        public int order;
        public string gloss;
        public string animation_clip;
        public bool is_fingerspelled;
        public float estimated_duration_sec;
        public string animation_url;
        public bool has_pose_animation;
        public string sign_status; // "approved", "clip_fallback", "no_sign"
    }

    [Serializable]
    private class BackendResponse
    {
        public string original_text;
        public string[] glosses;
        public bool reordered;
        public AnimationCueData[] animation_sequence;
    }

    #endregion
}
