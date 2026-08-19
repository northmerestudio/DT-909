from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

MODEL_ID = "Qwen/Qwen3.5-9B"
WD14_MODEL_ID = "SmilingWolf/wd-swinv2-tagger-v3"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
THREADS = max(1, (os.cpu_count() or 1) // 2)

# Retry only when the model output is invalid or incomplete.
MAX_OBSERVATION_ATTEMPTS = 5

# Broad enough to help Qwen, but do not feed demographic guesses back into it.
WD14_THRESHOLD = 0.35
WD14_MAX_TAGS = 45

SEX_VALUES = {"unknown", "female", "male"}
AGE_VALUES = {"unknown", "young adult", "mature adult", "elderly adult"}
HAIR_LENGTH_VALUES = {"unknown", "bald", "shaved", "short", "chin-length", "shoulder-length", "medium-length", "long", "waist-length"}
HAIR_TEXTURE_VALUES = {"unknown", "straight", "wavy", "curly", "coily", "kinky"}
HAIR_COLOR_VALUES = {"unknown", "black", "dark-brown", "brown", "dark-blonde", "blonde", "auburn", "red", "gray", "white"}
SKIN_TONE_VALUES = {"unknown", "fair", "light", "olive", "tan", "brown", "dark"}
EYE_COLOR_VALUES = {"unknown", "amber", "black", "blue", "brown", "gray", "green", "hazel"}
FIGURE_VALUES = {"unknown", "thin", "average", "curvy", "fat"}
CHEST_SIZE_VALUES = {"unknown", "small", "medium", "large"}
AGE_ORDER = ["young adult", "mature adult", "elderly adult"]
HAIR_LENGTH_ORDER = [
    "short", "chin-length", "shoulder-length",
    "medium-length", "long", "waist-length",
]
HAIR_TEXTURE_ORDER = ["straight", "wavy", "curly", "coily", "kinky"]
SKIN_TONE_ORDER = ["fair", "light", "olive", "tan", "brown", "dark"]
FIGURE_ORDER = ["thin", "average", "curvy", "fat"]
CHEST_SIZE_ORDER = ["small", "medium", "large"]
# Natural dark-to-light hair colors form a useful continuum. Red/auburn and
# gray/white are separate families rather than points on that same scale.
HAIR_COLOR_FAMILIES = [
    ["black", "dark-brown", "brown", "dark-blonde", "blonde"],
    ["auburn", "red"],
    ["gray", "white"],
]


@dataclass(frozen=True, slots=True)
class WD14Tag:
    name: str
    probability: float
    category: int


@dataclass(slots=True)
class Observation:
    medium: str = "photograph"
    sex: str = "unknown"
    age_range: str = "unknown"
    hair_length: str = "unknown"
    hair_texture: str = "unknown"
    hair_color: str = "unknown"
    skin_tone: str = "unknown"
    eye_color: str = "unknown"
    figure: str = "unknown"
    chest_size: str = "unknown"

    face_clear: bool = False
    hair_visible: bool = False
    skin_visible: bool = False
    eyes_clear: bool = False
    body_build_visible: bool = False
    chest_visible: bool = False

    # These are full factual sentences about this frame only. They must not
    # contain stable identity attributes; those are rendered deterministically.
    subject_action: str = ""
    expression_gaze: str = ""
    clothing_accessories: str = ""
    hands_feet: str = ""
    scene: str = ""
    camera: str = ""
    lighting: str = ""

    skin_features: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Consensus:
    sex: str = "unknown"
    age_range: str = "unknown"
    hair_length: str = "unknown"
    hair_texture: str = "unknown"
    hair_color: str = "unknown"
    skin_tone: str = "unknown"
    eye_color: str = "unknown"
    figure: str = "unknown"
    chest_size: str = "unknown"



@dataclass(frozen=True, slots=True)
class WD14Runtime:
    session: Any
    names: list[str]
    categories: np.ndarray
    size: int


@dataclass(frozen=True, slots=True)
class QwenRuntime:
    model: Any
    processor: Any


def find_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------- WD14 -----------------------------------------

def load_wd14() -> WD14Runtime:
    import onnxruntime as ort

    model_path = hf_hub_download(WD14_MODEL_ID, "model.onnx")
    labels_path = hf_hub_download(WD14_MODEL_ID, "selected_tags.csv")

    names: list[str] = []
    categories: list[int] = []
    with Path(labels_path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            names.append(row["name"])
            categories.append(int(row["category"]))

    available_providers = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers
        else ["CPUExecutionProvider"]
    )
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    shape = session.get_inputs()[0].shape
    if len(shape) != 4 or shape[1] != shape[2] or not isinstance(shape[1], int):
        raise ValueError(f"Unexpected WD14 input shape: {shape}")
    print(f"WD14 provider: {session.get_providers()[0]}")
    return WD14Runtime(session, names, np.asarray(categories, dtype=np.int16), int(shape[1]))


def load_oriented_rgb(path: Path) -> Image.Image:
    """Load one frame with EXIF orientation applied, capped at 1 MP and aligned to 32 px."""
    with Image.open(path) as source:
        source.seek(0)
        image = ImageOps.exif_transpose(source).convert("RGB")

        max_pixels = 1_000_000
        step = 32

        width, height = image.size
        pixels = width * height

        # Downscale to the 1 MP budget first.
        if pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            width = round(width * scale)
            height = round(height * scale)

        # Align dimensions to Qwen's 32-pixel spatial grid.
        width = max(step, round(width / step) * step)
        height = max(step, round(height / step) * step)

        # Rounding to 32 can push us slightly above the pixel budget.
        # If so, step the larger dimension down until we are <= 1 MP.
        while width * height > max_pixels:
            if width >= height:
                width -= step
            else:
                height -= step

        if image.size != (width, height):
            image = image.resize(
                (width, height),
                Image.Resampling.LANCZOS,
            )

        return image

def prepare_wd14_image(path: Path, size: int) -> np.ndarray:
    # Match WD14's reference preprocessing: apply EXIF orientation, composite
    # transparency onto white, square-pad on white, resize with bicubic, then BGR.
    with Image.open(path) as source:
        source.seek(0)
        image = ImageOps.exif_transpose(source)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")

    w, h = image.size
    side = max(w, h)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(image, ((side - w) // 2, (side - h) // 2))
    if side != size:
        square = square.resize((size, size), Image.Resampling.BICUBIC)
    rgb = np.asarray(square, dtype=np.float32)
    return np.expand_dims(np.ascontiguousarray(rgb[:, :, ::-1]), axis=0)


def wd14_tags(path: Path, runtime: WD14Runtime) -> list[WD14Tag]:
    image = prepare_wd14_image(path, runtime.size)
    inp = runtime.session.get_inputs()[0]
    out = runtime.session.get_outputs()[0]
    probs = runtime.session.run([out.name], {inp.name: image})[0][0]

    tags: list[WD14Tag] = []
    for i, prob in enumerate(probs):
        if runtime.categories[i] != 0 or float(prob) < WD14_THRESHOLD:
            continue
        readable = runtime.names[i].replace("_", " ").strip()
        tags.append(WD14Tag(readable, float(prob), int(runtime.categories[i])))
    tags.sort(key=lambda x: x.probability, reverse=True)
    return tags[:WD14_MAX_TAGS]


def format_wd14(tags: list[WD14Tag]) -> str:
    return ", ".join(f"{t.name} [{t.probability:.2f}]" for t in tags)


# --------------------------- Qwen -----------------------------------------

def load_qwen() -> QwenRuntime:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen in this script.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading: {MODEL_ID}")
    print(f"dtype: {dtype}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=dtype,
    ).eval()

    return QwenRuntime(model, processor)


OBSERVATION_PROMPT = r"""
Analyze this image for dataset captioning.

Return ONLY one valid JSON object matching the schema below.
No markdown, explanation, or extra text.
Return every key exactly once, even when the value is "unknown", false, [], or "".

GENERAL RULES

- Describe only facts visually supported by this image.
- Use "unknown" for unsupported structured attributes and an empty string for unsupported free-text fields.
- Do not infer names, ethnicity/race, nationality, exact age, exact measurements, or off-frame details.
- Keep descriptions neutral and suitable for general workplace use.
- Do not repeat stable attributes such as age, hair, skin tone, eye color, figure, or chest size inside free-text fields; those belong only in their structured fields.
- For anatomical left/right, use the subject's own perspective. If the side is uncertain, avoid left/right.
- Do not describe the same fact in multiple free-text fields.
- Free-text fields should be concise factual sentences, with at most one sentence per field.
- In free-text fields, refer to the main person as "The subject" when grammatically the subject, "the subject" when grammatically the object, and "The subject's"/"the subject's" for possession. Do not use personal pronouns; Python renders them later.

VISIBILITY RULES

- sex describes the visibly presented adult subject only.
  Use "unknown" when it is not visually clear.

- face_clear=true only when the face is clear enough to estimate age_range.
  Otherwise age_range="unknown".

- hair_visible=true only when scalp hair is sufficiently visible. Only scalp hair counts; facial and body hair do not.
  Otherwise hair_length, hair_texture, and hair_color must be "unknown". When hair_visible=true, judge hair_length, hair_texture, and hair_color independently; any individual attribute may still be "unknown".

- skin_visible=true only when enough skin is visible to judge skin_tone.
  Otherwise skin_tone="unknown".

- eyes_clear=true only when the irises are clear enough to judge eye_color. Judge the iris itself, not pupil darkness, reflections, eyelashes, or eyelid shadow; open eyes alone do not make eyes_clear=true.
  Otherwise eye_color="unknown" and do not infer gaze from the eyes.

- body_build_visible=true only when enough of the torso/body silhouette is visible to judge figure.
  Otherwise figure="unknown".

- chest_visible=true only when the chest area is clearly visible enough to estimate size.
  Otherwise chest_size="unknown".

FIELD RULES

subject_action:
Write one complete sentence beginning with "The subject" when grammatically appropriate.
Describe overall pose, body orientation, major limb posture, and head/face direction.
Do not describe eye gaze here; eye gaze belongs only in expression_gaze.
Do not describe clothing or detailed hand/foot positions here.

expression_gaze:
Describe visible facial expression.
Describe gaze only when eyes_clear=true and iris direction is visually reliable.

clothing_accessories:
Describe visible clothing, accessories, makeup, piercings, and tattoos.
Do not infer clothing outside the crop.

hands_feet:
Describe directly visible hand and foot positions.
Do not infer contact with an object or surface unless the contact is visibly clear.
When side is uncertain, use neutral wording such as "one hand", "the other hand", "one foot", or "the other foot" instead of guessing left/right.

scene:
Describe only background objects, environment, and clearly visible spatial relationships.
Do not repeat the subject's pose or appearance details.
Leave empty if there is nothing useful to describe.

camera:
Describe clearly supported viewpoint, angle, crop, and framing.
Do not guess an angle or shot-size label when ambiguous.
Use "full-body" only when the subject is continuously visible from head through both feet; if either foot is outside the frame, do not use "full-body".

lighting:
Describe visible illumination and shadows.
Do not infer a light source unless the source itself is visible, and do not speculate with words such as "likely" or "probably" about an unseen source.

skin_features:
Include only visible localized skin marks such as freckles, moles, or scars.
Entries must be concise noun phrases suitable after the word "include", such as "freckles on the shoulders" or "a mole on the left hip", not sentences.
Do not include tattoos, piercings, clothing, jewelry, or general body structure.

ENUMS

sex:
unknown | female | male

age_range:
unknown | young adult | mature adult | elderly adult

hair_length:
unknown | bald | shaved | short | chin-length | shoulder-length | medium-length | long | waist-length

hair_texture:
unknown | straight | wavy | curly | coily | kinky

hair_color:
unknown | black | dark-brown | brown | dark-blonde | blonde | auburn | red | gray | white

skin_tone:
unknown | fair | light | olive | tan | brown | dark

eye_color:
unknown | amber | black | blue | brown | gray | green | hazel

figure:
unknown | thin | average | curvy | fat

chest_size:
unknown | small | medium | large

OUTPUT SCHEMA

{
  "sex": "unknown",
  "age_range": "unknown",
  "hair_length": "unknown",
  "hair_texture": "unknown",
  "hair_color": "unknown",
  "skin_tone": "unknown",
  "eye_color": "unknown",
  "figure": "unknown",
  "chest_size": "unknown",

  "face_clear": false,
  "hair_visible": false,
  "skin_visible": false,
  "eyes_clear": false,
  "body_build_visible": false,
  "chest_visible": false,

  "skin_features": [],

  "subject_action": "",
  "expression_gaze": "",
  "clothing_accessories": "",
  "hands_feet": "",
  "scene": "",
  "camera": "",
  "lighting": ""
}
""".strip()


def _extract_model_text(text: str) -> str:
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return text


def _json_object(text: str) -> dict[str, Any]:
    text = _extract_model_text(text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model did not return a JSON object")
    return json.loads(text[start : end + 1])


def _enum(value: Any, allowed: set[str]) -> str:
    value = str(value or "unknown").strip().casefold().replace("_", "-")
    aliases = {"dark brown": "dark-brown", "dark blonde": "dark-blonde", "grey": "gray"}
    value = aliases.get(value, value)
    return value if value in allowed else "unknown"


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() == "true"
    return False

def _sentence(value: Any) -> str:
    s = " ".join(str(value or "").split()).strip()
    if not s:
        return ""
    # One tiny sanitation layer, not an English rewriter.
    s = re.sub(r"^(?:[-*]\s*)+", "", s)
    if s[-1] not in ".!?":
        s += "."
    return s


def parse_observation(data: dict[str, Any]) -> Observation:
    obs = Observation(
        medium=" ".join(str(data.get("medium") or "photograph").split()),
        sex=_enum(data.get("sex"), SEX_VALUES),
        age_range=_enum(data.get("age_range"), AGE_VALUES),
        hair_length=_enum(data.get("hair_length"), HAIR_LENGTH_VALUES),
        hair_texture=_enum(data.get("hair_texture"), HAIR_TEXTURE_VALUES),
        hair_color=_enum(data.get("hair_color"), HAIR_COLOR_VALUES),
        skin_tone=_enum(data.get("skin_tone"), SKIN_TONE_VALUES),
        eye_color=_enum(data.get("eye_color"), EYE_COLOR_VALUES),
        figure=_enum(data.get("figure"), FIGURE_VALUES),
        chest_size=_enum(data.get("chest_size"), CHEST_SIZE_VALUES),
        face_clear=_bool(data.get("face_clear", False)),
        hair_visible=_bool(data.get("hair_visible", False)),
        skin_visible=_bool(data.get("skin_visible", False)),
        eyes_clear=_bool(data.get("eyes_clear", False)),
        body_build_visible=_bool(data.get("body_build_visible", False)),
        chest_visible=_bool(data.get("chest_visible", False)),
        subject_action=_sentence(data.get("subject_action")),
        expression_gaze=_sentence(data.get("expression_gaze")),
        clothing_accessories=_sentence(data.get("clothing_accessories")),
        hands_feet=_sentence(data.get("hands_feet")),
        scene=_sentence(data.get("scene")),
        camera=_sentence(data.get("camera")),
        lighting=_sentence(data.get("lighting")),
    )
    features = data.get("skin_features", [])
    if isinstance(features, list):
        cleaned_features: list[str] = []
        seen_features: set[str] = set()
        for value in features:
            feature = " ".join(str(value).split()).strip()
            key = feature.casefold()
            if not feature or key in seen_features:
                continue
            seen_features.add(key)
            cleaned_features.append(feature)
        obs.skin_features = cleaned_features[:4]

    # Visibility is authoritative. This prevents a model from returning a value
    # while simultaneously saying the relevant feature is not visible.
    if not obs.face_clear:
        obs.age_range = "unknown"
    if not obs.hair_visible:
        obs.hair_length = obs.hair_texture = obs.hair_color = "unknown"
    if not obs.skin_visible:
        obs.skin_tone = "unknown"
    if not obs.eyes_clear:
        obs.eye_color = "unknown"
    if not obs.body_build_visible:
        obs.figure = "unknown"
    if not obs.chest_visible:
        obs.chest_size = "unknown"

    return obs


def observation_detail_score(obs: Observation) -> int:
    """Count useful populated observation slots without rewarding verbosity."""
    score = sum(
        value != "unknown"
        for value in (
            obs.sex,
            obs.age_range,
            obs.hair_length,
            obs.hair_texture,
            obs.hair_color,
            obs.skin_tone,
            obs.eye_color,
            obs.figure,
            obs.chest_size,
        )
    )
    score += sum(
        bool(text)
        for text in (
            obs.subject_action,
            obs.expression_gaze,
            obs.clothing_accessories,
            obs.hands_feet,
            obs.scene,
            obs.camera,
            obs.lighting,
        )
    )
    score += min(len(obs.skin_features), 4)
    return int(score)


def analyze_image(path: Path, tags: list[WD14Tag], qwen: QwenRuntime) -> Observation:
    model, processor = qwen.model, qwen.processor
    tag_text = format_wd14(tags)
    prompt = OBSERVATION_PROMPT + (
        f"\n\nNon-demographic WD14 hints (untrusted):\n{tag_text}"
        if tag_text else ""
    )

    oriented_image = load_oriented_rgb(path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": oriented_image},
            {"type": "text", "text": prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )

    input_device = model.get_input_embeddings().weight.device
    inputs = inputs.to(input_device)
    input_tokens = inputs["input_ids"].shape[-1]

    last_error: Exception | None = None

    try:
        for attempt in range(1, MAX_OBSERVATION_ATTEMPTS + 1):
            started = time.perf_counter()

            try:
                generation = dict(
                    max_new_tokens=1280,
                    do_sample=attempt > 1,
                    use_cache=True,
                )
                if attempt > 1:
                    generation.update(temperature=0.7, top_p=0.8, top_k=20)

                print(
                    f"Observation attempt {attempt}/{MAX_OBSERVATION_ATTEMPTS}...",
                    end="",
                    flush=True,
                )

                with torch.inference_mode():
                    ids = model.generate(**inputs, **generation)

                elapsed = time.perf_counter() - started

                generated_ids = ids[0, input_tokens:].detach().cpu()
                raw = processor.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                del ids, generated_ids

                obs = parse_observation(_json_object(raw))
                if not obs.subject_action:
                    raise ValueError("required subject_action missing")

                score = observation_detail_score(obs)
                print(
                    f" done in {elapsed:.1f}s "
                    f"(attributes={score})"
                )
                return obs

            except Exception as exc:
                elapsed = time.perf_counter() - started
                last_error = exc
                print(
                    f" failed after {elapsed:.1f}s: {exc}"
                )

        raise ValueError(
            f"Qwen produced no valid observation after "
            f"{MAX_OBSERVATION_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    finally:
        del inputs
        oriented_image.close()


# ---------------------- consensus + rendering -----------------------------

def winner(values: Sequence[str], minimum_support: int = 2, minimum_share: float = 0.55) -> str:
    """Plurality winner for genuinely categorical attributes."""
    known_values = [value for value in values if value != "unknown"]
    if not known_values:
        return "unknown"

    value, support = Counter(known_values).most_common(1)[0]
    share = support / len(known_values)
    if support < minimum_support or share < minimum_share:
        return "unknown"
    return value


def neighborhood_winner(
    values: Sequence[str],
    order: Sequence[str],
    *,
    adjacent_weight: float = 0.50,
    near_weight: float = 0.15,
    minimum_support: int = 2,
    minimum_direct_share: float = 0.10,
) -> str:
    """Choose a representative value on an ordered scale.

    Votes support their own category fully, an adjacent category partly, and a
    category two steps away weakly. The winner must still have meaningful direct
    support, so a scarcely observed middle label cannot win merely by sitting
    between two opposing clusters.
    """
    known_values = [value for value in values if value in order]
    if not known_values:
        return "unknown"

    counts = Counter(known_values)
    minimum_direct = max(
        minimum_support,
        math.ceil(len(known_values) * minimum_direct_share),
    )

    scores: dict[str, float] = {}
    for candidate_index, candidate in enumerate(order):
        score = 0.0
        for value, count in counts.items():
            distance = abs(candidate_index - order.index(value))
            if distance == 0:
                weight = 1.0
            elif distance == 1:
                weight = adjacent_weight
            elif distance == 2:
                weight = near_weight
            else:
                weight = 0.0
            score += count * weight
        scores[candidate] = score

    ranked = sorted(
        order,
        key=lambda value: (-scores[value], -counts[value], order.index(value)),
    )
    for value in ranked:
        if counts[value] >= minimum_direct:
            return value
    return "unknown"


def hair_length_winner(values: Sequence[str]) -> str:
    known_values = [value for value in values if value != "unknown"]
    if not known_values:
        return "unknown"

    counts = Counter(known_values)
    scalp_state_support = counts["bald"] + counts["shaved"]

    # Bald/shaved are scalp states, not neighboring hair lengths. Only let that
    # family win when it represents a clear share of the usable observations.
    if scalp_state_support / len(known_values) >= 0.50:
        return winner(
            [value for value in known_values if value in {"bald", "shaved"}],
            minimum_share=0.55,
        )

    return neighborhood_winner(known_values, HAIR_LENGTH_ORDER)


def hair_color_winner(values: Sequence[str]) -> str:
    known_values = [value for value in values if value != "unknown"]
    if not known_values:
        return "unknown"

    counts = Counter(known_values)
    family_scores = [
        sum(counts[value] for value in family)
        for family in HAIR_COLOR_FAMILIES
    ]
    best_family_index = max(range(len(family_scores)), key=family_scores.__getitem__)
    family = HAIR_COLOR_FAMILIES[best_family_index]
    family_support = family_scores[best_family_index]

    # If no color family clearly dominates, do not force unrelated hues onto a
    # single ordered scale.
    if family_support / len(known_values) < 0.60:
        return winner(known_values, minimum_share=0.55)

    family_values = [value for value in known_values if value in family]

    # On the natural black->blonde axis, two strong labels separated by exactly
    # one missing intermediate label can indicate a stable in-between color.
    # Example: repeated brown + blonde observations can reasonably support
    # dark-blonde when lighting makes frames fall on either side of that boundary.
    if len(family) >= 3:
        family_counts = Counter(family_values)
        ranked = family_counts.most_common()
        if len(ranked) >= 2:
            first, first_count = ranked[0]
            second, second_count = ranked[1]
            i, j = sorted((family.index(first), family.index(second)))
            combined = first_count + second_count
            weaker = min(first_count, second_count)
            middle = family[i + 1] if j - i == 2 else None

            if (
                middle is not None
                and combined / len(family_values) >= 0.70
                and weaker / len(family_values) >= 0.20
                and family_counts[middle] / len(family_values) <= 0.10
            ):
                return middle

    return neighborhood_winner(family_values, family)


def chest_size_winner(observations: Sequence[Observation]) -> str:
    values = [
        o.chest_size
        for o in observations
        if o.chest_visible and o.chest_size != "unknown"
    ]
    if not values:
        return "unknown"

    # Keep consensus purely visual; do not use age as a tie-break for size.
    return neighborhood_winner(values, CHEST_SIZE_ORDER)



def usable_eye_observations(
    observations: Sequence[Observation],
) -> list[Observation]:
    """Return eye-color observations that are usable for consensus."""
    usable = [
        obs
        for obs in observations
        if obs.eyes_clear and obs.eye_color != "unknown"
    ]
    if not usable:
        return []

    bad_light_terms = (
        "colored light", "colored lighting", "neon",
        "red light", "blue light", "green light", "purple light",
        "strong shadow", "harsh shadow", "dramatic shadow",
    )
    cleaner = [
        obs
        for obs in usable
        if not any(term in obs.lighting.casefold() for term in bad_light_terms)
    ]
    return cleaner if len(cleaner) >= 4 else usable


def eye_color_winner(observations: Sequence[Observation]) -> str:
    """Return eye color only when usable observations agree convincingly."""
    chosen = usable_eye_observations(observations)
    if not chosen:
        return "unknown"

    counts = Counter(obs.eye_color for obs in chosen)
    ranked = counts.most_common(2)
    color, support = ranked[0]
    total = len(chosen)
    runner_up = ranked[1][1] if len(ranked) > 1 else 0

    if support < 2 or support / total < 0.60:
        return "unknown"
    if (support - runner_up) / total < 0.15:
        return "unknown"
    return color


def _visible_values(
    observations: Sequence[Observation],
    value_attr: str,
    visibility_attr: str | None = None,
) -> list[str]:
    """Collect an attribute, optionally restricted to visible observations."""
    return [
        getattr(obs, value_attr)
        for obs in observations
        if visibility_attr is None or getattr(obs, visibility_attr)
    ]


def build_consensus(observations: Sequence[Observation]) -> Consensus:
    age_values = _visible_values(observations, "age_range", "face_clear")
    hair_length_values = _visible_values(observations, "hair_length", "hair_visible")
    hair_texture_values = _visible_values(observations, "hair_texture", "hair_visible")
    hair_color_values = _visible_values(observations, "hair_color", "hair_visible")
    skin_tone_values = _visible_values(observations, "skin_tone", "skin_visible")
    figure_values = _visible_values(observations, "figure", "body_build_visible")

    age_range = neighborhood_winner(age_values, AGE_ORDER)

    return Consensus(
        sex=winner(_visible_values(observations, "sex")),
        age_range=age_range,
        hair_length=hair_length_winner(hair_length_values),
        hair_texture=neighborhood_winner(hair_texture_values, HAIR_TEXTURE_ORDER),
        hair_color=hair_color_winner(hair_color_values),
        skin_tone=neighborhood_winner(skin_tone_values, SKIN_TONE_ORDER),
        # Eye colors are categorical. A weak plurality should remain unknown
        # rather than being treated like an ordered shade scale.
        eye_color=eye_color_winner(observations),
        figure=neighborhood_winner(figure_values, FIGURE_ORDER),
        chest_size=chest_size_winner(observations),
    )


def _direct_consensus_share(
    observations: Sequence[Observation],
    value_attr: str,
    consensus_value: str,
    visibility_attr: str | None = None,
) -> float:
    """Return direct support for a consensus label among usable observations."""
    values = (
        [obs.eye_color for obs in usable_eye_observations(observations)]
        if value_attr == "eye_color"
        else [
            value
            for value in _visible_values(observations, value_attr, visibility_attr)
            if value != "unknown"
        ]
    )
    if not values or consensus_value == "unknown":
        return 0.0
    return values.count(consensus_value) / len(values)


def apply_consensus(
    obs: Observation,
    consensus: Consensus,
    observations: Sequence[Observation],
) -> Observation:
    """Return a copy with only well-supported stable fields normalized."""
    updates: dict[str, Any] = {
        "skin_features": list(obs.skin_features),
    }

    # Thresholds control propagation, not whether a dataset-level consensus can
    # be displayed. Weak or derived consensus therefore does not overwrite good
    # per-image observations.
    propagation = {
        "sex": (None, 0.80),
        "age_range": ("face_clear", 0.70),
        "hair_length": ("hair_visible", 0.70),
        "hair_texture": ("hair_visible", 0.65),
        "hair_color": ("hair_visible", 0.65),
        "skin_tone": ("skin_visible", 0.65),
        "eye_color": ("eyes_clear", 0.65),
        "figure": ("body_build_visible", 0.65),
        "chest_size": ("chest_visible", 0.65),
    }

    for field_name, (visibility_field, minimum_share) in propagation.items():
        value = getattr(consensus, field_name)
        if value == "unknown":
            continue
        if visibility_field and not getattr(obs, visibility_field):
            continue
        if field_name.startswith("hair_") and getattr(obs, field_name) == "unknown":
            continue

        share = _direct_consensus_share(
            observations,
            field_name,
            value,
            visibility_field,
        )
        if share >= minimum_share:
            updates[field_name] = value

    return replace(obs, **updates)


def subject_words(obs: Observation) -> tuple[str, str, str, str]:
    """Return noun, subject pronoun, object pronoun, possessive determiner."""
    if obs.sex == "female":
        return "woman", "she", "her", "her"
    if obs.sex == "male":
        return "man", "he", "him", "his"
    return "person", "they", "them", "their"


def apply_subject_pronouns(text: str, obs: Observation) -> str:
    """Replace prompt-safe subject placeholders with grammatical pronouns."""
    if not text:
        return ""

    _, subject, object_, possessive = subject_words(obs)

    def preserve_case(word: str, original: str) -> str:
        return word.capitalize() if original[0].isupper() else word

    # Possessive first.
    text = re.sub(
        r"\bthe subject['’]s\b",
        lambda m: preserve_case(possessive, m.group(0)),
        text,
        flags=re.IGNORECASE,
    )

    # Prompt convention: capitalized = grammatical subject, lowercase = object.
    text = re.sub(r"\bThe subject\b", subject.capitalize(), text)
    text = re.sub(r"\bthe subject\b", object_, text)

    # Small fallback if the model leaks pronouns despite the prompt.
    if obs.sex != "unknown":
        text = re.sub(
            r"\btheir\b",
            lambda m: preserve_case(possessive, m.group(0)),
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bthey\b",
            lambda m: preserve_case(subject, m.group(0)),
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bthem\b",
            lambda m: preserve_case(object_, m.group(0)),
            text,
            flags=re.IGNORECASE,
        )

    # Fix the common plural-verb leak after singular pronoun substitution.
    if obs.sex in {"female", "male"}:
        text = re.sub(r"\b([Ss]he|[Hh]e)\s+are\b", r"\1 is", text)

    return text

def hair_phrase(obs: Observation) -> str:
    parts = [x for x in (obs.hair_length, obs.hair_texture, obs.hair_color) if x != "unknown"]
    return " ".join(parts) + " hair" if parts else ""


def identity_sentence(obs: Observation) -> str:
    bits: list[str] = []
    hp = hair_phrase(obs)
    if hp:
        bits.append(hp)
    if obs.skin_tone != "unknown":
        bits.append(f"{obs.skin_tone} skin")
    if obs.eye_color != "unknown":
        bits.append(f"{obs.eye_color} eyes")
    if obs.figure != "unknown":
        bits.append(
            f"an {obs.figure} figure"
            if obs.figure == "average"
            else f"a {obs.figure} figure"
        )

    _, subject, _, possessive = subject_words(obs)
    subject_cap = subject.capitalize()
    possessive_cap = possessive.capitalize()

    sentences: list[str] = []
    if obs.age_range != "unknown":
        sentences.append(f"{subject_cap} is a {obs.age_range}.")
    if bits:
        if len(bits) == 1:
            joined = bits[0]
        elif len(bits) == 2:
            joined = f"{bits[0]} and {bits[1]}"
        else:
            joined = ", ".join(bits[:-1]) + f", and {bits[-1]}"
        sentences.append(f"{subject_cap} has {joined}.")
    # Report chest size only when the frame supports that neutral visual estimate.
    if obs.chest_visible and obs.chest_size != "unknown":
        sentences.append(f"{possessive_cap} chest is {obs.chest_size}.")
    if obs.skin_features:
        sentences.append(
            f"{possessive_cap} visible localized skin features include "
            + ", ".join(obs.skin_features)
            + "."
        )
    return " ".join(sentences)


def _content_words(text: str) -> set[str]:
    """Return normalized content words for conservative deduplication."""
    words = re.findall(r"[a-z0-9]+", text.casefold())
    ignored = {
        "a", "an", "and", "are", "as", "at", "her", "his",
        "in", "is", "of", "on", "she", "he", "the", "their",
        "they", "to", "with",
    }
    return {word for word in words if word not in ignored}


def _near_duplicate(text: str, previous: Sequence[str]) -> bool:
    """Detect only very similar repeated sentences/parts."""
    words = _content_words(text)
    if len(words) < 4:
        return False
    for other in previous:
        other_words = _content_words(other)
        if not other_words:
            continue
        overlap = len(words & other_words) / len(words | other_words)
        if overlap >= 0.85:
            return True
    return False


def render_caption(obs: Observation) -> str:
    noun, _, _, _ = subject_words(obs)
    parts = [
        f"This {obs.medium} shows a {noun}.",
        apply_subject_pronouns(obs.subject_action, obs),
        identity_sentence(obs),
        apply_subject_pronouns(obs.expression_gaze, obs),
        apply_subject_pronouns(obs.clothing_accessories, obs),
        apply_subject_pronouns(obs.hands_feet, obs),
        apply_subject_pronouns(obs.scene, obs),
        apply_subject_pronouns(obs.camera, obs),
        apply_subject_pronouns(obs.lighting, obs),
    ]
    clean_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        part = part.strip() if part else ""
        if not part:
            continue
        key = part.casefold()
        if key in seen or _near_duplicate(part, clean_parts):
            continue
        seen.add(key)
        clean_parts.append(part)

    text = " ".join(clean_parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _consensus_summary_line(
    label: str,
    consensus_value: str,
    values: list[str],
    *,
    title_value: bool = False,
) -> str:
    """Format one consensus row with agreement and optional vote spread."""
    known_values = [value for value in values if value != "unknown"]
    counts = Counter(known_values)
    eligible = len(known_values)

    display_value = (
        consensus_value.title()
        if title_value and consensus_value != "unknown"
        else ("Unknown" if consensus_value == "unknown" else consensus_value)
    )

    if not eligible:
        agreement = "[0/0, 0%]"
        votes = ""
    else:
        if consensus_value != "unknown":
            support = counts[consensus_value]
            if support:
                share = support / eligible
                agreement = f"[{support}/{eligible}, {share:.0%}]"
            else:
                # Some ordered/family-aware consensus values can be inferred
                # between strongly supported neighboring categories.
                agreement = f"[derived from {eligible} votes]"
        else:
            support = max(counts.values())
            share = support / eligible
            agreement = f"[{support}/{eligible}, {share:.0%}]"

        # Only print the vote spread when observations disagree.
        if len(counts) > 1:
            ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            votes = "   votes: " + ", ".join(
                f"{value}={count}" for value, count in ordered
            )
        else:
            votes = ""

    # Keep values aligned while allowing longer labels such as Chest size.
    return f"{label + ':':<22}{display_value:<24}{agreement}{votes}"


def format_summary(c: Consensus, observations: Sequence[Observation]) -> str:
    rows = (
        ("Sex", "sex", None, False),
        ("Age range", "age_range", "face_clear", True),
        ("Hair length", "hair_length", "hair_visible", False),
        ("Hair texture", "hair_texture", "hair_visible", False),
        ("Hair color", "hair_color", "hair_visible", False),
        ("Skin tone", "skin_tone", "skin_visible", False),
        ("Eye color", "eye_color", "eyes_clear", False),
        ("Figure", "figure", "body_build_visible", False),
        (
            "Chest size",
            "chest_size",
            "chest_visible",
            False,
        ),
    )

    lines: list[str] = []
    for label, value_attr, visibility_attr, title_value in rows:
        values = (
            [obs.eye_color for obs in usable_eye_observations(observations)]
            if value_attr == "eye_color"
            else _visible_values(observations, value_attr, visibility_attr)
        )
        lines.append(
            _consensus_summary_line(
                label,
                getattr(c, value_attr),
                values,
                title_value=title_value,
            )
        )
    return "\n".join(lines)



def analyze_directory(
    images: list[Path],
    wd14: WD14Runtime,
    qwen: QwenRuntime,
) -> tuple[dict[Path, Observation], int]:
    observations: dict[Path, Observation] = {}
    failures = 0

    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path.name}")
        try:
            tags = wd14_tags(path, wd14)
            observations[path] = analyze_image(path, tags, qwen)
        except Exception as exc:
            failures += 1
            print(f"FAILED: {exc}")

    return observations, failures

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption all supported images in a directory.",
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing images to caption.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(THREADS)
    args = parse_args(argv)
    directory = args.directory.expanduser().resolve()

    if not directory.is_dir():
        print(f"Directory not found: {directory}")
        return 2

    images = find_images(directory)
    if not images:
        print(f"No images found in {directory}")
        return 2


    # Fail before expensive work if outputs would collide.
    outputs = [path.with_suffix(".txt") for path in images]
    if len(set(outputs)) != len(outputs):
        print("Two image files would map to the same .txt output name.")
        return 2

    print(f"Images: {len(images)}")
    wd14 = load_wd14()
    qwen = load_qwen()

    try:
        observations, failures = analyze_directory(images, wd14, qwen)

        if not observations:
            print("No images were successfully analyzed.")
            return 1

        observation_values = list(observations.values())
        consensus = build_consensus(observation_values)
        print("\n--- Consensus ---")
        print(format_summary(consensus, observation_values))

        for path, observation in observations.items():
            caption = render_caption(apply_consensus(observation, consensus, observation_values))
            output_path = path.with_suffix(".txt")
            atomic_write(output_path, caption)
            print(f"Saved {output_path.name} ({len(caption.split())} words)")

        print(f"\nDone: {len(observations)} saved, {failures} analysis failures.")
        return int(failures > 0)
    finally:
        # Release both CUDA users: Qwen/PyTorch and WD14/ONNX Runtime.
        # torch.cuda.empty_cache() only affects PyTorch's allocator; it cannot
        # release memory owned by the ONNX Runtime CUDA execution provider.
        try:
            del qwen
        except UnboundLocalError:
            pass
        try:
            del wd14
        except UnboundLocalError:
            pass
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
