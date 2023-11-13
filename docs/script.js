const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let coord = { x: 0, y: 0 };

    const labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    const displayElement = document.getElementById('displayText');


    canvas.addEventListener("mousedown", start);
    document.addEventListener("mouseup", stop);
    // window.addEventListener("resize", resize);

    // resize();

    // function resize() {
    //   ctx.canvas.width = window.innerWidth;
    //   ctx.canvas.height = window.innerHeight;
    // }

    canvas.addEventListener('touchstart', start);
    document.addEventListener('touchend', stop);

    canvas.addEventListener('touchmove', function (event) {event.preventDefault();});


    function reposition(event) {
      coord.x = (event.touches ? event.touches[0].clientX : event.clientX) - canvas.getBoundingClientRect().left;
      coord.y = (event.touches ? event.touches[0].clientY : event.clientY) - canvas.getBoundingClientRect().top;;
    }
    function start(event) {
      document.addEventListener("mousemove", draw);
      document.addEventListener('touchmove', draw);
      reposition(event);
    }
    function stop() {
      document.removeEventListener("mousemove", draw);
      document.removeEventListener("touchmove", draw);
    }
    function draw(event) {
      ctx.beginPath();
      ctx.lineWidth = 30;
      ctx.lineCap = "round";
      ctx.strokeStyle = "#FFF";
      ctx.moveTo(coord.x, coord.y);
      reposition(event);
      ctx.lineTo(coord.x, coord.y);
      ctx.stroke();
    }

    function resetCanvas() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    function Recognize() {
      predict();
    }
    function RecognizeReset() {
      predict();
      resetCanvas();
    }

    async function loadModel() {
      session = await ort.InferenceSession.create('./EMNIST.onnx')
    }
    loadModel()

    async function predict() {
      // const session = await ort.InferenceSession.create('./EMNIST.onnx')

      let tmpCanvas = document.createElement('canvas');
      tmpCanvas.width = canvas.width;
      tmpCanvas.height = canvas.height;
      let tmpCtx = tmpCanvas.getContext('2d');
      tmpCtx.filter = 'blur(6px)';
      tmpCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height);

      let predictCanvas = document.getElementById('canvas_test');
      predictCanvas.width = 28;
      predictCanvas.height = 28;
      let predictCtx = predictCanvas.getContext('2d', { willReadFrequently: true });
      predictCtx.drawImage(tmpCanvas, 0, 0, 28, 28);
      let imgData = predictCtx.getImageData(0, 0, 28, 28).data;

      // console.log("=== imgData", imgData)

      let data = new Float32Array(28 * 28);
      for (let i = 0; i < imgData.length; i++) {
        data[i] = imgData[i * 4 + 3]
      }
      // console.log("=== input", data)

      let transposedData = new Float32Array(28 * 28)
      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          transposedData[x * 28 + y] = data[y * 28 + x]
        }
      }

      const input = new ort.Tensor('float32', transposedData, [1, 1, 28, 28])
      const result = await session.run({ 'input': input })
      const logits = result.output.data
      const probas = softmax(logits)
      // console.log("=== Result", probas)
      let maxIndex = indexOfMax(probas)
      let label = labels[maxIndex]
      console.log("=== label", label)
      displayElement.textContent = "the model predicted : " + label + " with probability : " + probas[maxIndex].toFixed(2);
    }

    const getImgData = () => {
      const data = []
      for (let i = 0; i < 1 * 1 * 28 * 28; ++i) {
        data.push(0)
      }
      return data
    }

    const softmax = (data) => {
      const exps = data.map((value) => Math.exp(value))
      const sumExps = exps.reduce((acc, val) => acc + val)
      return exps.map((exp) => exp / sumExps)
    }

    function indexOfMax(arr) {
      if (arr.length === 0) {
        return -1;
      }

      var max = arr[0];
      var maxIndex = 0;

      for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
          maxIndex = i;
          max = arr[i];
        }
      }

      return maxIndex;
    }

    // const main = async () => {
    //   session = await ort.InferenceSession.create('./EMNIST.onnx')
    //   const data = Float32Array.from(getImgData()).map((pixel) => (pixel - 0.1307) / 0.3081)
    //   const input = new ort.Tensor('float32', data, [1, 1, 28, 28])
    //   const result = await session.run({ 'input': input })
    //   const logits = result.output.data
    //   const probas = softmax(logits)
    //   console.log("=== Result", probas)
    // }
    // main()