# UI asset provenance

## `morphagent_ui/resources/morphagent_hero.png`

- **Mode:** New image generated from text; no source or reference image was supplied.
- **Generator:** Built-in OpenAI image-generation tool available in the development environment.
- **Purpose:** Original MorphAgent home artwork communicating resolution-aware microscopy and interpretable evidence; it does not reproduce a Nellie asset or a manuscript figure.
- **Repository path:** `morphagent_ui/resources/morphagent_hero.png`
- **Prompt:**

  > Create an original landscape scientific hero image for a microscopy feature-discovery desktop application. Use a dark ink/navy background and a diagonal transition from a softly blurred wide-field cell view on the left to a sharply resolved cell on the right. In the resolved half, show cyan reticular networks, violet bundles and nucleus structure, sparse coral puncta, and a few precise evidence callout nodes/lines. The style should feel like premium scientific visualization, restrained and technically credible, with strong contrast and generous edge-safe composition. No text, letters, logos, watermark, robot, humanoid agent, generic DNA helix, copied paper panel, or existing product branding.

- **Downstream changes:** Cropped/scaled at runtime only. The repository stores the generated PNG without overlays or borrowed graphical elements.
