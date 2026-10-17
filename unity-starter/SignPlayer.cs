using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Plays sequences of MediaPipe bone-rotation animation clips on a Humanoid avatar.
/// Overrides Animator poses in LateUpdate() and smoothly blends (slerps) between consecutive signs.
/// </summary>
[RequireComponent(typeof(Animator))]
public class SignPlayer : MonoBehaviour
{
    [Header("Components")]
    [SerializeField] private Animator animator;
    [SerializeField] private PoseJsonLoader jsonLoader;

    [Header("Playback Settings")]
    [Tooltip("Crossfade blend time between consecutive signs in seconds")]
    [Range(0.05f, 0.5f)]
    [SerializeField] private float blendDuration = 0.2f;

    [Tooltip("Rest pause between signs in seconds")]
    [Range(0.0f, 1.0f)]
    [SerializeField] private float pauseBetweenSigns = 0.15f;

    [Header("Test Sequence")]
    [SerializeField] private string[] testSequence = new string[] { "AYUBOWAN", "STHUTHI" };

    // Active state
    private Dictionary<HumanBodyBones, Transform> boneTransformCache = new Dictionary<HumanBodyBones, Transform>();
    private Dictionary<HumanBodyBones, Quaternion> currentBoneRotations = new Dictionary<HumanBodyBones, Quaternion>();
    private Coroutine activeSequenceRoutine;

    public bool IsPlaying => activeSequenceRoutine != null;

    private void Awake()
    {
        if (animator == null) animator = GetComponent<Animator>();
        if (jsonLoader == null) jsonLoader = GetComponent<PoseJsonLoader>() ?? gameObject.AddComponent<PoseJsonLoader>();

        CacheHumanoidBones();
    }

    private void Start()
    {
        // Optional quick-start test trigger via Inspector or keyboard
    }

    private void CacheHumanoidBones()
    {
        if (animator == null || !animator.isHuman) return;

        foreach (HumanBodyBones bone in Enum.GetValues(typeof(HumanBodyBones)))
        {
            if (bone == HumanBodyBones.LastBone) continue;
            Transform t = animator.GetBoneTransform(bone);
            if (t != null)
            {
                boneTransformCache[bone] = t;
            }
        }
        Debug.Log($"[SignPlayer] Cached {boneTransformCache.Count} humanoid bone transforms.");
    }

    private void LateUpdate()
    {
        // Override bone local rotations on top of Animator in LateUpdate
        if (currentBoneRotations.Count > 0)
        {
            foreach (var kvp in currentBoneRotations)
            {
                if (boneTransformCache.TryGetValue(kvp.Key, out Transform boneTransform) && boneTransform != null)
                {
                    boneTransform.localRotation = kvp.Value;
                }
            }
        }
    }

    /// <summary>
    /// Play a sequence of sign glosses sequentially with smooth blending between them.
    /// </summary>
    public void PlayGlossSequence(IList<string> glosses, Action onComplete = null)
    {
        if (activeSequenceRoutine != null)
        {
            StopCoroutine(activeSequenceRoutine);
        }
        activeSequenceRoutine = StartCoroutine(PlaybackSequenceRoutine(glosses, onComplete));
    }

    /// <summary>
    /// Public helper for testing in Unity Inspector.
    /// </summary>
    [ContextMenu("Play Test Sequence")]
    public void PlayTestSequence()
    {
        if (testSequence != null && testSequence.Length > 0)
        {
            PlayGlossSequence(testSequence);
        }
    }

    private IEnumerator PlaybackSequenceRoutine(IList<string> glosses, Action onComplete)
    {
        for (int i = 0; i < glosses.Count; i++)
        {
            string gloss = glosses[i];
            PoseAnimationData.AnimationClipData clip = null;
            bool loadFailed = false;

            yield return jsonLoader.LoadAnimation(
                gloss,
                onSuccess: loadedClip => clip = loadedClip,
                onError: errMsg => {
                    Debug.LogWarning($"[SignPlayer] {errMsg}");
                    loadFailed = true;
                }
            );

            if (loadFailed || clip == null || clip.frames.Count == 0)
            {
                continue;
            }

            // Play clip frames
            yield return PlaySingleClip(clip);

            // Optional crossfade / pause before next clip
            if (i < glosses.Count - 1 && pauseBetweenSigns > 0)
            {
                yield return new WaitForSeconds(pauseBetweenSigns);
            }
        }

        activeSequenceRoutine = null;
        onComplete?.Invoke();
    }

    /// <summary>
    /// Play a single sign by gloss name using Pose JSON animation data.
    /// Returns true via onComplete callback if successfully loaded and played.
    /// </summary>
    public IEnumerator PlaySingleSign(string gloss, Action<bool> onComplete = null)
    {
        PoseAnimationData.AnimationClipData clip = null;
        bool loadFailed = false;

        yield return jsonLoader.LoadAnimation(
            gloss,
            onSuccess: loadedClip => clip = loadedClip,
            onError: errMsg => {
                Debug.LogWarning($"[SignPlayer] {errMsg}");
                loadFailed = true;
            }
        );

        if (loadFailed || clip == null || clip.frames.Count == 0)
        {
            onComplete?.Invoke(false);
            yield break;
        }

        yield return PlaySingleClip(clip);
        onComplete?.Invoke(true);
    }

    public IEnumerator PlaySingleClip(PoseAnimationData.AnimationClipData clip)
    {
        float frameTime = 1f / Mathf.Max(1f, clip.fps);
        int totalFrames = clip.frames.Count;

        // If we already have active bone poses, blend smoothly into the first frame of this clip
        if (currentBoneRotations.Count > 0 && blendDuration > 0)
        {
            var startPose = new Dictionary<HumanBodyBones, Quaternion>(currentBoneRotations);
            var targetPose = clip.frames[0];
            float blendElapsed = 0f;

            while (blendElapsed < blendDuration)
            {
                blendElapsed += Time.deltaTime;
                float t = Mathf.Clamp01(blendElapsed / blendDuration);

                foreach (var kvp in targetPose)
                {
                    Quaternion qStart = startPose.ContainsKey(kvp.Key) ? startPose[kvp.Key] : kvp.Value;
                    currentBoneRotations[kvp.Key] = Quaternion.Slerp(qStart, kvp.Value, t);
                }
                yield return null;
            }
        }

        // Advance frames according to clip FPS
        for (int f = 0; f < totalFrames; f++)
        {
            var frameBones = clip.frames[f];
            foreach (var kvp in frameBones)
            {
                currentBoneRotations[kvp.Key] = kvp.Value;
            }
            yield return new WaitForSeconds(frameTime);
        }
    }
}
