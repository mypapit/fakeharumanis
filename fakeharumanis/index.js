const CLASS_LABEL = Array("non-scary", "scary");

const fileInput = document.getElementById("file-input");
const image = document.getElementById("image");
const result = document.getElementById("prediction");
const explain = document.getElementById("fuzzyexplain");

fileInput.addEventListener("change", getImage);

// Async loading

model = "";
(async () => {
  console.log("before start");

  document.getElementById("loader").style.display = "block";

  model = await tf.loadLayersModel("model/model.json");
  //model.summary();
  console.log("after start");
  document.getElementById("loader").style.display = "none";

  // Remove loading class from body
  document.body.classList.remove("loading");

  // When user uploads a new image, display the new image on the webpage
  fileInput.addEventListener("change", getImage);
})();

function getImage() {
  // Check if an image has been found in the input
  if (!fileInput.files[0]) throw new Error("Image not found");
  const file = fileInput.files[0];

  // Get the data url form the image
  const reader = new FileReader();

  // When reader is ready display image
  reader.onload = function(event) {
    // Ge the data url
    const dataUrl = event.target.result;

    // Create image object
    const imageElement = new Image();
    imageElement.src = dataUrl;

    // When image object is loaded
    imageElement.onload = function() {
      // Set <img /> attributes
      image.setAttribute("src", this.src);
      image.setAttribute("height", this.height);
      image.setAttribute("width", this.width);

      document.getElementById("loader").style.display = "block";
      console.log("height " + this.height);

      // Classify image
      classifyImage();
    };

    // Add the image-loaded class to the body
    document.body.classList.add("image-loaded");
  };

  // Get data URL
  reader.readAsDataURL(file);
}

fileInput.addEventListener("change", getImage);

function classifyImage() {

  console.log("akan predict")
  console.log(image)
  tensor = tf.browser
    .fromPixels(image)
    .resizeNearestNeighbor([128, 128])
    .toFloat().expandDims().sub(127.5).div(127.5);


    ktensor=tf.tensor([[1, 2], [3, 4]]);

  (async () => {

    predictions = await model.predict(ktensor).print();
    console.log("tengah predict");

    console.log(predictions);

    });

}

function toggleBlur(obj) {
  obj.classList.toggle("blur");
}

function ddx() {
  if (navigator.share) {
    navigator
      .share({
        title: "ScaryNet",
        text: "Compact Neural network for detecting scary images",
        url: window.location.href
      })
      .then(() => {
        //alert('Thanks for sharing!');
      })
      .catch(err => {
        alert("Couldn't share because " + err.message);
      });
  } else {
    //alert('Web share not supported, please use compatible device!');
  }
}

function dprobability() {
  result.classList.toggle("showprobability");
}
