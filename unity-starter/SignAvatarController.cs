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
    [SerializeField] private float defaultClipDuration = 1.2f;

    [Header("Quick Start Test (Day 3 Hardcoded)")]
    [SerializeField] private string[] testGlossSequence = new string[] { "Salute", "Shooting_Arrow" };

    private Coroutine activePlaybackRoutine;

    private void Awake()
    {
        if (animator == null)
        {
            animator = GetComponent<Animator>();
        }
    }

    /// <summary>
    /// Day 2 Deliverable: Play a single sign animation clip by gloss name with cross-fading.
    /// </summary>
    public void PlaySign(string glossName)
    {
        if (animator == null)
        {
            Debug.LogError("[SignAvatarController] Animator component missing!");
            return;
        }

        Debug.Log($"[SignAvatarController] Playing sign: {glossName}");
        // Try playing directly or with space/underscore normalized
        try
        {
            animator.CrossFadeInFixedTime(glossName, transitionDuration);
        }
        catch (Exception)
        {
            string altName = glossName.Contains("_") ? glossName.Replace("_", " ") : glossName.Replace(" ", "_");
            animator.CrossFadeInFixedTime(altName, transitionDuration);
        }
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
        foreach (string gloss in glosses)
        {
            PlaySign(gloss);
            yield return new WaitForSeconds(defaultClipDuration);
        }

        // Return to idle state if configured in animator
        PlaySign("Idle");
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
                Debug.LogError($"[SignAvatarController] Translation API error: {request.error}\nResponse: {request.downloadHandler.text}");
            }
            else
            {
                string jsonResponse = request.downloadHandler.text;
                Debug.Log($"[SignAvatarController] Translation received: {jsonResponse}");

                BackendResponse parsed = JsonUtility.FromJson<BackendResponse>(jsonResponse);
                if (parsed != null && parsed.glosses != null && parsed.glosses.Length > 0)
                {
                    PlaySequence(parsed.glosses);
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
