using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// SignAvatarController controls sign language animation playback on a rigged humanoid avatar.
/// Supports both Day 2 (single animation trigger) and Day 3 (sequential sentence playback + FastAPI bridge).
/// </summary>
public class SignAvatarController : MonoBehaviour
{
    [Header("Components")]
    [SerializeField] private Animator animator;

    [Header("Backend API Configuration")]
    [SerializeField] private string backendUrl = "http://127.0.0.1:8000/api/v1/translate";
    [SerializeField] private float transitionDuration = 0.25f;
    [SerializeField] private float defaultClipDuration = 1.6f;

    [Header("Quick Start Test (Day 3 Hardcoded)")]
    [SerializeField] private string[] testGlossSequence = new string[] { "Salute", "Shooting_Arrow" };

    [Header("Live Status")]
    public string currentStatus = "Ready";

    private Coroutine activePlaybackRoutine;
    private Vector3 initialPosition;
    private Quaternion initialRotation;

    private void Awake()
    {
        if (animator == null)
        {
            animator = GetComponent<Animator>();
        }

        // CRITICAL: Disable Root Motion so animations (like Shooting Arrow) do not
        // physically rotate the avatar sideways or change the camera angle!
        if (animator != null)
        {
            animator.applyRootMotion = false;
        }

        initialPosition = transform.position;
        initialRotation = transform.rotation;
    }

    private void LateUpdate()
    {
        // Enforce pinned transform in case an animation clip has unbaked root transforms
        if (activePlaybackRoutine != null && !animator.applyRootMotion)
        {
            transform.position = initialPosition;
            transform.rotation = initialRotation;
        }
    }

    /// <summary>
    /// Searches for a matching animation state in layer 0 across casing conventions and common aliases.
    /// Returns the exact state name if found, or null otherwise.
    /// </summary>
    public string FindMatchingState(string name)
    {
        if (animator == null || string.IsNullOrWhiteSpace(name)) return null;

        // Strip fingerspelling hash if present (e.g. "#AND" -> "AND")
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

        // Semantic aliases for our Mixamo clips
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

        // Check if any candidate exists in the Animator's Base Layer (Layer 0)
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

    /// <summary>
    /// Day 2 Deliverable: Play a single sign animation clip by gloss name with cross-fading.
    /// Returns true if played, false if state wasn't found.
    /// </summary>
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
            Debug.Log($"[SignAvatarController] Playing state: '{matchedState}' (from gloss: '{glossName}')");
            animator.CrossFadeInFixedTime(matchedState, transitionDuration);
            return true;
        }

        Debug.LogWarning($"[SignAvatarController] Animation state for '{glossName}' not found in Animator. Make sure you added a state named '{glossName}' in AvatarController!");
        return false;
    }

    /// <summary>
    /// Day 3 Deliverable: Play the hardcoded test sequence of animations one after another.
    /// </summary>
    [ContextMenu("Play Hardcoded Test Sequence")]
    public void PlayTestSequence()
    {
        PlaySequence(testGlossSequence);
    }

    /// <summary>
    /// Play a sequence of gloss animation names sequentially.
    /// </summary>
    public void PlaySequence(IEnumerable<string> glosses)
    {
        if (activePlaybackRoutine != null)
        {
            StopCoroutine(activePlaybackRoutine);
        }
        activePlaybackRoutine = StartCoroutine(PlaySequenceCoroutine(glosses));
    }

    private IEnumerator PlaySequenceCoroutine(IEnumerable<string> glosses)
    {
        currentStatus = "Starting sequence...";
        foreach (string gloss in glosses)
        {
            bool played = PlaySign(gloss);
            if (played)
            {
                yield return new WaitForSeconds(defaultClipDuration);
            }
        }

        // Return to idle/starting pose after sentence finishes so avatar doesn't stay frozen
        string idleState = FindMatchingState("Idle") ?? FindMatchingState("Salute");
        if (idleState != null)
        {
            animator.CrossFadeInFixedTime(idleState, transitionDuration);
        }

        transform.position = initialPosition;
        transform.rotation = initialRotation;
        currentStatus = "Sequence finished (Ready)";
        activePlaybackRoutine = null;
    }

    /// <summary>
    /// Translates an English sentence via the FastAPI backend and plays the returned animation sequence.
    /// </summary>
    public void TranslateAndPlay(string sentence)
    {
        StartCoroutine(SendTranslationRequest(sentence));
    }

    private IEnumerator SendTranslationRequest(string sentence)
    {
        currentStatus = "Connecting to Python Backend...";
        string jsonPayload = $"{{\"text\": \"{EscapeJson(sentence)}\", \"enable_reordering\": true, \"fingerspell_unknown\": true}}";

        using (UnityWebRequest request = new UnityWebRequest(backendUrl, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonPayload);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                currentStatus = "API Error (Check backend server)";
                Debug.LogError($"[SignAvatarController] Translation API error: {request.error}\nResponse: {request.downloadHandler.text}");
            }
            else
            {
                string jsonResponse = request.downloadHandler.text;
                Debug.Log($"[SignAvatarController] Translation received: {jsonResponse}");

                BackendResponse parsed = JsonUtility.FromJson<BackendResponse>(jsonResponse);
                if (parsed != null && parsed.glosses != null && parsed.glosses.Length > 0)
                {
                    currentStatus = $"Translated: {string.Join(" → ", parsed.glosses)}";
                    PlaySequence(parsed.glosses);
                }
                else
                {
                    currentStatus = "No signs returned";
                }
            }
        }
    }

    private string EscapeJson(string str)
    {
        return str.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r");
    }

    [Serializable]
    private class BackendResponse
    {
        public string original_text;
        public string[] glosses;
        public bool reordered;
    }
}
