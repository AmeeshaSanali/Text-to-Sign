using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Simple UI helper that connects a Legacy/TextMeshPro InputField and Button to SignAvatarController.
/// Also provides an immediate OnGUI overlay if Canvas UI is not yet set up!
/// </summary>
public class SignAvatarUI : MonoBehaviour
{
    [SerializeField] private SignAvatarController avatarController;
    private string sentenceInput = "I am going to the store";

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
        GUILayout.BeginArea(new Rect(20, 20, 360, 220), GUI.skin.box);
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
        if (GUILayout.Button("Play Hardcoded Test ['I', 'GO', 'STORE']"))
        {
            if (avatarController != null)
            {
                avatarController.PlayTestSequence();
            }
        }

        GUILayout.Space(8);
        GUILayout.Label("Quick Single Signs:");
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("I")) avatarController.PlaySign("I");
        if (GUILayout.Button("GO")) avatarController.PlaySign("GO");
        if (GUILayout.Button("STORE")) avatarController.PlaySign("STORE");
        if (GUILayout.Button("Idle")) avatarController.PlaySign("Idle");
        GUILayout.EndHorizontal();

        GUILayout.EndArea();
    }
}
