using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Interactive OnGUI interface for live typed-word sign language playback.
/// Detects finished words (space, punctuation, or Enter) and queues signs in real-time.
/// Visually renders currently playing sign, upcoming queue, and unknown/skipped signs.
/// </summary>
public class SignAvatarUI : MonoBehaviour
{
    [SerializeField] private SignAvatarController avatarController;
    [SerializeField] private bool autoTriggerOnTypedWord = true;

    private string sentenceInput = "hello go blue";
    private string lastReportedInput = "";
    private Vector2 queueScrollPos;

    private void Awake()
    {
        if (avatarController == null)
        {
            avatarController = GetComponent<SignAvatarController>();
        }
    }

    private void OnGUI()
    {
        GUILayout.BeginArea(new Rect(20, 20, 480, 480), GUI.skin.box);
        GUILayout.Label("<b><size=14>SLSL Avatar — Real-Time Typed Signer</size></b>");

        // Status Header
        string statusText = avatarController != null ? avatarController.currentStatus : "Ready";
        GUILayout.Label($"<color=cyan>Status: {statusText}</color>");

        // Currently Playing Sign Banner
        if (avatarController != null && avatarController.CurrentPlayingSign != null)
        {
            var active = avatarController.CurrentPlayingSign;
            string mode = active.hasPoseAnimation ? "3D Pose" : "Clip";
            GUILayout.Label($"<color=#00FF88><b>▶ NOW SIGNING: {active.gloss} [{mode}]</b></color>");
        }
        else
        {
            GUILayout.Label("<color=#94A3B8>Idle (Waiting for input)</color>");
        }

        GUILayout.Space(6);
        GUILayout.Label("<b>Live Typed Input (signs trigger on space / punctuation / Enter):</b>");

        // Input Text Field
        GUI.SetNextControlName("SentenceField");
        string newText = GUILayout.TextField(sentenceInput);
        if (newText != sentenceInput)
        {
            sentenceInput = newText;
            if (autoTriggerOnTypedWord)
            {
                CheckWordBoundary(sentenceInput);
            }
        }

        // Detect Enter key press in the input field
        Event e = Event.current;
        if (e.isKey && e.keyCode == KeyCode.Return && e.type == EventType.KeyDown)
        {
            if (avatarController != null && !string.IsNullOrWhiteSpace(sentenceInput))
            {
                avatarController.OnTypedWordFinished(sentenceInput);
                lastReportedInput = sentenceInput.Trim();
            }
        }

        GUILayout.Space(6);
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Send / Retranslate"))
        {
            if (avatarController != null)
            {
                avatarController.OnTypedWordFinished(sentenceInput);
                lastReportedInput = sentenceInput.Trim();
            }
        }
        if (GUILayout.Button("Clear Queue & Reset"))
        {
            if (avatarController != null)
            {
                avatarController.ClearQueue();
                sentenceInput = "";
                lastReportedInput = "";
            }
        }
        GUILayout.EndHorizontal();

        GUILayout.Space(8);
        GUILayout.Label("<b>Live Playback Queue:</b>");

        // Queue Display Area
        List<SignAvatarController.QueuedSign> queuedList = avatarController != null ? avatarController.QueuedSigns : new List<SignAvatarController.QueuedSign>();

        GUILayout.BeginVertical(GUI.skin.box, GUILayout.Height(100));
        queueScrollPos = GUILayout.BeginScrollView(queueScrollPos);

        if (queuedList.Count == 0 && (avatarController == null || avatarController.CurrentPlayingSign == null))
        {
            GUILayout.Label("<color=grey><i>Queue is empty. Type words above to enqueue signs.</i></color>");
        }
        else
        {
            if (avatarController != null && avatarController.CurrentPlayingSign != null)
            {
                GUILayout.Label($"<color=#00FF88>● [Playing] {avatarController.CurrentPlayingSign.gloss}</color>");
            }

            for (int i = 0; i < queuedList.Count; i++)
            {
                var item = queuedList[i];
                if (item.isUnknown)
                {
                    GUILayout.Label($"<color=grey>○ [{i + 1}] {item.gloss} (Unknown / No Sign)</color>");
                }
                else
                {
                    string modeTag = item.hasPoseAnimation ? "Pose" : "Clip";
                    GUILayout.Label($"<color=yellow>○ [{i + 1}] {item.gloss} [{modeTag}]</color>");
                }
            }
        }

        GUILayout.EndScrollView();
        GUILayout.EndVertical();

        GUILayout.Space(6);
        GUILayout.Label("<b>Quick Test Samples:</b>");
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("hello go blue")) SetAndTrigger("hello go blue");
        if (GUILayout.Button("today mother eat")) SetAndTrigger("today mother eat");
        if (GUILayout.Button("thank you")) SetAndTrigger("thank you");
        GUILayout.EndHorizontal();

        GUILayout.Space(4);
        GUILayout.Label("<b>Clip Mode Fallback Triggers:</b>");
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Salute")) avatarController?.PlaySign("Salute");
        if (GUILayout.Button("Shoot Arrow")) avatarController?.PlaySign("Shooting_Arrow");
        if (GUILayout.Button("Hardcoded [Salute -> Shoot]")) avatarController?.PlayTestSequence();
        GUILayout.EndHorizontal();

        GUILayout.EndArea();
    }

    private void CheckWordBoundary(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return;

        char lastChar = text[text.Length - 1];
        bool isBoundary = char.IsWhiteSpace(lastChar) || char.IsPunctuation(lastChar);

        string trimmed = text.Trim();
        if (isBoundary && trimmed != lastReportedInput && !string.IsNullOrEmpty(trimmed))
        {
            lastReportedInput = trimmed;
            if (avatarController != null)
            {
                avatarController.OnTypedWordFinished(trimmed);
            }
        }
    }

    private void SetAndTrigger(string sample)
    {
        sentenceInput = sample;
        lastReportedInput = sample.Trim();
        if (avatarController != null)
        {
            avatarController.ClearQueue();
            avatarController.OnTypedWordFinished(sample);
        }
    }
}
