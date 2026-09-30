using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Simple UI helper that connects a Legacy/TextMeshPro InputField and Button to SignAvatarController.
/// Also provides an immediate OnGUI overlay if Canvas UI is not yet set up!
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

    // Instant zero-setup test UI rendered in the Game view
    private void OnGUI()
    {
        GUILayout.BeginArea(new Rect(20, 20, 380, 230), GUI.skin.box);
        GUILayout.Label("<b>SLSL Avatar — Quick Start Tester</b>");

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
