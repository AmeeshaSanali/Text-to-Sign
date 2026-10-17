using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Data structures for deserializing MediaPipe retargeted animation JSON files.
/// </summary>
public static class PoseAnimationData
{
    [Serializable]
    public class BoneRotation
    {
        public string bone;
        public float x;
        public float y;
        public float z;
        public float w;

        public Quaternion ToQuaternion() => new Quaternion(x, y, z, w);
    }

    [Serializable]
    public class Frame
    {
        public List<BoneRotation> bones = new List<BoneRotation>();
    }

    public class AnimationClipData
    {
        public string gloss;
        public float fps = 30f;
        public int frameCount;
        public List<Dictionary<HumanBodyBones, Quaternion>> frames = new List<Dictionary<HumanBodyBones, Quaternion>>();
    }
}

/// <summary>
/// Loads MediaPipe bone-rotation JSON files at runtime from local StreamingAssets or HTTP REST endpoints.
/// WebGL-compatible via UnityWebRequest.
/// </summary>
public class PoseJsonLoader : MonoBehaviour
{
    [Header("Configuration")]
    [Tooltip("Base URL or directory for fetching animation JSONs")]
    [SerializeField] private string baseAnimationUrl = "http://127.0.0.1:8000/animations/slsl/";

    /// <summary>
    /// Coroutine to fetch and parse an animation JSON by gloss name.
    /// </summary>
    public IEnumerator LoadAnimation(string gloss, Action<PoseAnimationData.AnimationClipData> onSuccess, Action<string> onError)
    {
        string cleanGloss = gloss.Trim().ToUpper().Replace("-", "_");
        string url = baseAnimationUrl.EndsWith("/") ? $"{baseAnimationUrl}{cleanGloss}.json" : $"{baseAnimationUrl}/{cleanGloss}.json";

        // Check local StreamingAssets fallback paths
        string local1 = System.IO.Path.Combine(Application.streamingAssetsPath, $"{cleanGloss}.json");
        string local2 = System.IO.Path.Combine(Application.streamingAssetsPath, "animations", "slsl", $"{cleanGloss}.json");

        // 1. Check local StreamingAssets file on disk first for zero-latency offline loading
        string chosenLocal = System.IO.File.Exists(local1) ? local1 : (System.IO.File.Exists(local2) ? local2 : null);
        if (chosenLocal != null)
        {
            try
            {
                string jsonText = System.IO.File.ReadAllText(chosenLocal);
                PoseAnimationData.AnimationClipData clip = ParseAnimationJson(jsonText, cleanGloss);
                if (clip != null && clip.frames.Count > 0)
                {
                    Debug.Log($"[PoseJsonLoader] Successfully loaded '{cleanGloss}' ({clip.frames.Count} frames) from local StreamingAssets.");
                    onSuccess?.Invoke(clip);
                    yield break;
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[PoseJsonLoader] Error reading local file {chosenLocal}: {ex.Message}");
            }
        }

        // 2. If not found locally, fetch via HTTP network request
        if (url.StartsWith("http://", StringComparison.OrdinalIgnoreCase) || url.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
        {
            Debug.Log($"[PoseJsonLoader] Fetching animation for '{cleanGloss}' from {url}");
            using (UnityWebRequest req = UnityWebRequest.Get(url))
            {
                req.timeout = 3; // 3-second timeout to prevent stalling
                yield return req.SendWebRequest();

                if (req.result == UnityWebRequest.Result.Success)
                {
                    string jsonText = req.downloadHandler.text;
                    PoseAnimationData.AnimationClipData clip = ParseAnimationJson(jsonText, cleanGloss);
                    if (clip != null && clip.frames.Count > 0)
                    {
                        onSuccess?.Invoke(clip);
                        yield break;
                    }
                }
                else
                {
                    Debug.LogWarning($"[PoseJsonLoader] HTTP fetch failed ({req.error})");
                }
            }
        }

        onError?.Invoke($"Failed to find animation for {cleanGloss} (checked HTTP and local StreamingAssets)");
    }

    /// <summary>
    /// Parses JSON string containing bone rotations per frame.
    /// Supports flexible format: { "gloss": "NAME", "fps": 30, "frames": [ { "bones": { "RightUpperArm": [x,y,z,w] } } ] }
    /// </summary>
    public static PoseAnimationData.AnimationClipData ParseAnimationJson(string jsonText, string fallbackGloss = "UNKNOWN")
    {
        try
        {
            var clip = new PoseAnimationData.AnimationClipData();
            clip.gloss = fallbackGloss;

            // Simple robust token scanner for bone rotations per frame
            // Avoids third-party dependencies so it works out of the box in standard Unity WebGL
            int framesArrayStart = jsonText.IndexOf("\"frames\"", StringComparison.OrdinalIgnoreCase);
            if (framesArrayStart == -1) return null;

            // Extract fps if present
            int fpsIdx = jsonText.IndexOf("\"fps\"", StringComparison.OrdinalIgnoreCase);
            if (fpsIdx != -1)
            {
                int colon = jsonText.IndexOf(':', fpsIdx);
                int comma = jsonText.IndexOf(',', colon);
                if (colon != -1 && comma != -1)
                {
                    string fpsStr = jsonText.Substring(colon + 1, comma - colon - 1).Trim();
                    if (float.TryParse(fpsStr, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float parsedFps))
                    {
                        clip.fps = parsedFps > 0 ? parsedFps : 30f;
                    }
                }
            }

            // Extract frames
            int openBracket = jsonText.IndexOf('[', framesArrayStart);
            if (openBracket == -1) return null;

            int current = openBracket + 1;
            while (current < jsonText.Length)
            {
                int frameObjStart = jsonText.IndexOf('{', current);
                if (frameObjStart == -1) break;

                int frameObjEnd = FindMatchingBrace(jsonText, frameObjStart);
                if (frameObjEnd == -1) break;

                string frameBlock = jsonText.Substring(frameObjStart, frameObjEnd - frameObjStart + 1);
                var frameBones = ParseBonesInFrameBlock(frameBlock);
                clip.frames.Add(frameBones);

                current = frameObjEnd + 1;
            }

            clip.frameCount = clip.frames.Count;
            return clip;
        }
        catch (Exception ex)
        {
            Debug.LogError($"[PoseJsonLoader] Exception parsing animation JSON: {ex.Message}");
            return null;
        }
    }

    private static Dictionary<HumanBodyBones, Quaternion> ParseBonesInFrameBlock(string frameJson)
    {
        var result = new Dictionary<HumanBodyBones, Quaternion>();
        int idx = 0;

        while (idx < frameJson.Length)
        {
            int colon = frameJson.IndexOf(':', idx);
            if (colon == -1) break;

            int quoteEnd = frameJson.LastIndexOf('"', colon);
            if (quoteEnd == -1) { idx = colon + 1; continue; }
            int quoteStart = frameJson.LastIndexOf('"', quoteEnd - 1);
            if (quoteStart == -1) { idx = colon + 1; continue; }

            string keyPart = frameJson.Substring(quoteStart + 1, quoteEnd - quoteStart - 1).Trim();

            int openBracket = frameJson.IndexOf('[', colon);
            if (openBracket == -1) { idx = colon + 1; continue; }

            // Guard against another key before bracket
            int nextColon = frameJson.IndexOf(':', colon + 1);
            if (nextColon != -1 && nextColon < openBracket) { idx = colon + 1; continue; }

            int closeBracket = frameJson.IndexOf(']', openBracket);
            if (closeBracket == -1) { idx = openBracket + 1; continue; }

            string numbersStr = frameJson.Substring(openBracket + 1, closeBracket - openBracket - 1);
            string[] parts = numbersStr.Split(new char[] { ',', '\n', '\r', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 4)
            {
                if (float.TryParse(parts[0], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float x) &&
                    float.TryParse(parts[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float y) &&
                    float.TryParse(parts[2], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float z) &&
                    float.TryParse(parts[3], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float w))
                {
                    if (Enum.TryParse<HumanBodyBones>(keyPart, true, out HumanBodyBones boneEnum))
                    {
                        result[boneEnum] = new Quaternion(x, y, z, w);
                    }
                }
            }

            idx = closeBracket + 1;
        }

        return result;
    }

    private static int FindMatchingBrace(string text, int openPos)
    {
        int count = 0;
        for (int i = openPos; i < text.Length; i++)
        {
            if (text[i] == '{') count++;
            else if (text[i] == '}')
            {
                count--;
                if (count == 0) return i;
            }
        }
        return -1;
    }
}
