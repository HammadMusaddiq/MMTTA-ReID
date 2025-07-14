<h2>📁 <code>configs/</code> — Multimodal Configuration Files</h2>

<p>This folder contains YAML configuration files for training and evaluating the <strong>MMTTA</strong> model on various datasets with support for multimodal inputs (<strong>RGB</strong>, <strong>IR</strong>, <strong>TI</strong>, <strong>Captions</strong>).</p>

<p>Each file corresponds to a specific dataset and modality combination, and includes:</p>
<ul>
  <li>🧱 <strong>Architecture details</strong></li>
  <li>🧪 <strong>Loss settings</strong></li>
  <li>🧬 <strong>Modality-specific preprocessing</strong></li>
</ul>

<hr>

<h3>🧠 Dataset: <strong>CeLReID</strong></h3>

<p>Configuration files for running <strong>ViT-based multimodal person re-identification</strong> on the <strong>CeLReID</strong> dataset.</p>

<table>
  <thead>
    <tr>
      <th>📄 <strong>Config File</strong></th>
      <th>🖼️ <strong>Modalities Used</strong></th>
      <th>📝 <strong>Description</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>vit_base.yml</code></td>
      <td><strong>RGB only</strong></td>
      <td>Baseline single-modality training with standard RGB images. No fusion modules are used.</td>
    </tr>
    <tr>
      <td><code>vit_base_2M-RGB+IR.yml</code></td>
      <td><strong>RGB + IR</strong></td>
      <td>Two-modality training using RGB and infrared inputs. <strong>Early fusion</strong> of modalities is supported.</td>
    </tr>
    <tr>
      <td><code>vit_base_3M-RGB+IR+TI.yml</code></td>
      <td><strong>RGB + IR + TI</strong></td>
      <td>Three-modality setup (RGB, IR, Thermal). The model uses <strong>dynamic fusion strategies</strong> to integrate visual modalities.</td>
    </tr>
    <tr>
      <td><code>vit_base_4M-RGB+IR+TI+1Caption-RGB.yml</code></td>
      <td><strong>RGB + IR + TI + 1 Caption (RGB)</strong></td>
      <td>Introduces a <strong>caption branch</strong> that uses only RGB image captions (e.g., generated via BLIP). <strong>BERT encodings</strong> are aligned using contrastive and triplet loss.</td>
    </tr>
    <tr>
      <td><code>vit_base_4M-RGB+IR+TI+All-Caption.yml</code></td>
      <td><strong>RGB + IR + TI + All Captions</strong></td>
      <td>All three captions (matching RGB, IR, and TI) are integrated. Caption vectors are <strong>BERT-embedded</strong> and fused with visual modalities.</td>
    </tr>
  </tbody>
</table>

<hr>

<h3>⚙️ <strong>Modality Integration Notes</strong></h3>
<ul>
  <li>✅ All configs support enabling/disabling fusion modules (<code>use_fusion_module</code>)</li>
  <li>🧠 Caption-based configs require <code>caption_path</code> and <code>caption_embedding</code> (e.g., <code>bert</code>)</li>
  <li>🔗 Fusion tokens (e.g., CLS) and loss functions are adapted per modality setup</li>
  <li>📂 Dataset paths and preprocessing are customized for CeLReID (<code>/mnt/data/...</code>)</li>
  <li>📌 <strong>NEW:</strong> All configs now include the <code>DATASETS.MULTI=True</code> and <code>DATASETS.NAMES=[rgbnt201, CelebReID]</code> flag to support combined multimodal training across multiple datasets. This allows the model to be trained on a **combination of datasets**, rather than a single dataset.</li>
</ul>

<hr>

<h3>▶️ <strong>Usage</strong></h3>
<p>To train a model using a specific config file:</p>

<pre><code>python train.py --config-file configs/vit_base_3M-RGB+IR+TI.yml
</code></pre>

<p><strong>Ensure the following:</strong></p>
<ul>
  <li>📁 Dataset folders and captions are correctly mounted</li>
  <li>🔧 Transformer model supports fusion and caption integration</li>
  <li>📉 Loss modules are enabled to handle caption/image alignment (if applicable)</li>
</ul>
