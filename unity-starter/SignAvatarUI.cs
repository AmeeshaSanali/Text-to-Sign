using UnityEngine;

/// <summary>
/// Simple UI helper that renders an instant OnGUI overlay in the Unity Game view.
/// Displays live translation status, single sign triggers, and API execution.
/// </summary>
public class SignAvatarUI : MonoBehaviour
{
    [SerializeField] private SignAvatarController avatarController;
    private string sentenceInput = "Salute and shoot the arrow";

    private void Awake()
    {
        if (avatarController == null)
        {
            avatarController = GetComponent<SignAvatarController>();
        }
    }

    private void OnGUI()
    {
        GUILayout.BeginArea(new Rect(20, 20, 400, 260), GUI.skin.box);
        GUILayout.Label("<b>SLSL Avatar — Quick Start Tester</b>");

        // Status display
        string status = avatarController != null ? avatarController.currentStatus : "Ready";
        GUILayout.Label($"<color=cyan>Status: {status}</color>");

        GUILayout.Space(6);
        GUILayout.Label("English Sentence:");
        sentenceInput = GUILayout.TextField(sentenceInput);

        GUILayout.Space(8);
        if (GUILayout.Button("Translate & Play (API Call)"))
        {
            if (avatarController != null)
            {
                avatarController.TranslateAndPlay(sentenceInput);
            }
        }

        GUILayout.Space(4);
        if (GUILayout.Button("Play Hardcoded Test ['Salute', 'Shooting_Arrow']"))
        {
            if (avatarController != null)
            {
                avatarController.PlayTestSequence();
            }
        }

        GUILayout.Space(8);
        GUILayout.Label("Single Sign Triggers:");
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Salute")) avatarController.PlaySign("Salute");
        if (GUILayout.Button("Shoot Arrow")) avatarController.PlaySign("Shooting_Arrow");
        GUILayout.EndHorizontal();

        GUILayout.EndArea();
    }
}
