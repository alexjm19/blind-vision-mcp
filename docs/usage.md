# Usage

## Tools

### vision_describe

Analyze an image with ultra-detailed text descriptions:

```
vision_describe(image_path="/path/to/photo.jpg")
vision_describe(image_path="https://example.com/photo.jpg", detail_level="brief")
vision_describe(image_path="/path/to/photo.jpg", detail_level="ultra", focus="defects")
```

**Parameters:**
- `image_path` (required): Absolute path or URL
- `detail_level`: `"brief"` | `"standard"` | `"ultra"` (default: `"ultra"`)
- `focus`: Optional area to emphasize (e.g. `"text"`, `"people"`, `"defects"`)

**Output structure** (ultra mode):
- DESCRIPCIÓN GENERAL
- COMPOSICIÓN
- SUJETOS PRINCIPALES
- COLORES E ILUMINACIÓN
- TEXTURAS Y MATERIALES
- TEXTO VISIBLE
- ANOMALÍAS
- CONTEXTO
- DETALLES TÉCNICOS

### vision_compare

Compare two images:

```
vision_compare(
  image_path_a="/path/to/photo_a.jpg",
  image_path_b="/path/to/photo_b.jpg"
)
vision_compare(
  image_path_a="/path/to/original.png",
  image_path_b="/path/to/edited.png",
  focus="color changes"
)
```

### image_generate

Generate an image from text:

```
image_generate(description="a mystical forest with glowing mushrooms at night")
image_generate(
  description="a futuristic city skyline",
  params={"width": 1280, "height": 720, "steps": 8}
)
image_generate(
  description="a portrait of a cyberpunk character with neon hair",
  params={"seed": 42, "override_prompt": true}
)
```

**Parameters:**
- `description` (required): Natural language description
- `output_path`: Save location (default: `outputs/generated_TIMESTAMP.png`)
- `params`: Optional dict:
  - `width`, `height`: Resolution (default: 1024x1024)
  - `steps`: Inference steps (default: 4)
  - `guidance_scale`: Classifier-free guidance (default: 0.0)
  - `seed`: For reproducible generation
  - `negative_prompt`: What to avoid
  - `override_prompt`: Skip prompt optimization (default: false)

### image_edit

Edit a masked zone of an image:

```
image_edit(
  image_path="/path/to/photo.jpg",
  mask_path="/path/to/mask.png",
  description="replace with a blooming cherry blossom tree"
)
```

**Mask format:** PNG with white pixels (255) in the editable zone,
black pixels (0) in the preserved zone.

### get_status

Check server health and VRAM usage:

```
get_status()
```
