
const fileInput = document.getElementById("file");
const preview   = document.getElementById("preview");
const outTxt    = document.getElementById("out");


let labels = [];
fetch("labels.json").then(r => r.json()).then(j => labels = j);

fileInput.onchange = async e => {
  const file = e.target.files[0];
  if (!file) return;


  preview.src    = URL.createObjectURL(file);
  preview.hidden = false;
  outTxt.textContent = "⏳ Inférence…";


  const bmp   = await createImageBitmap(file);
  const scale = 256 / Math.min(bmp.width, bmp.height);
  const w     = bmp.width  * scale;
  const h     = bmp.height * scale;

  const tmp  = new OffscreenCanvas(w, h);
  const tctx = tmp.getContext("2d");
  tctx.drawImage(bmp, 0, 0, w, h);

 
  const sx   = (w - 224) / 2;
  const sy   = (h - 224) / 2;
  const crop = tctx.getImageData(sx, sy, 224, 224).data;   


  const mean = [0.485, 0.456, 0.406];
  const std  = [0.229, 0.224, 0.225];
  const input = new Float32Array(1 * 3 * 224 * 224);

  for (let y = 0; y < 224; y++) {
    for (let x = 0; x < 224; x++) {
      const i   = (y * 224 + x) * 4;
      const idx =  y * 224 + x;

      const r = crop[i]   / 255;
      const g = crop[i+1] / 255;
      const b = crop[i+2] / 255;

      input[idx]                 = (r - mean[0]) / std[0];      // R
      input[224*224 + idx]       = (g - mean[1]) / std[1];      // G
      input[2*224*224 + idx]     = (b - mean[2]) / std[2];      // B
    }
  }


  const tensor  = new Tensor("float32", input, [1, 3, 224, 224]);
  const session = await InferenceSession.create("mobilenetv2.onnx");
  const output  = await session.run({ input: tensor });
  const scores  = output.output.data;


  let bestIdx = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[bestIdx]) bestIdx = i;
  }
  const label = labels[bestIdx];


  outTxt.textContent = label;
};
