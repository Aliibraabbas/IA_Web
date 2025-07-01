
const { InferenceSession, Tensor } = ort;

const canvas   = document.getElementById("canvas");
const ctx      = canvas.getContext("2d", { willReadFrequently: true });
const resTxt   = document.getElementById("result");
const clearBtn = document.getElementById("clear");
const predBtn  = document.getElementById("predict");

/* --- canvas principal --- */
ctx.fillStyle = "black";
ctx.fillRect(0, 0, 280, 280);
ctx.lineWidth   = 20;
ctx.lineCap     = "round";
ctx.strokeStyle = "white";

const smallCanvas = document.createElement("canvas");
smallCanvas.width  = 28;
smallCanvas.height = 28;
const smallCtx = smallCanvas.getContext("2d", { willReadFrequently: true });

/* --- gestion du tracé --- */
let drawing = false;

canvas.onmousedown = e => {
  drawing = true;
  const r = canvas.getBoundingClientRect();
  ctx.beginPath();
  ctx.moveTo(e.clientX - r.left, e.clientY - r.top);
};

canvas.onmouseup   = () => (drawing = false);

canvas.onmousemove = e => {
  if (!drawing) return;
  const r = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - r.left, e.clientY - r.top);
  ctx.stroke();
};

/* --- boutons --- */
clearBtn.onclick = () => {
  ctx.fillRect(0, 0, 280, 280);
  resTxt.textContent = "";
};

predBtn.onclick = async () => {
  resTxt.textContent = "⏳ Prédiction…";


  smallCtx.drawImage(canvas, 0, 0, 28, 28);
  const img = smallCtx.getImageData(0, 0, 28, 28).data;


  const input = new Float32Array(1 * 1 * 28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    const r = img[i * 4];         
    const v = r / 255;       
    input[i] = (v - 0.1307) / 0.3081;
  }


  const tensor  = new Tensor("float32", input, [1, 1, 28, 28]);
  const session = await InferenceSession.create("model_cnn.onnx");
  const output  = await session.run({ input: tensor });
  const scores  = output.output.data;
  const pred    = scores.indexOf(Math.max(...scores));

  resTxt.textContent = `Je pense que c’est un ${pred}`;
};
