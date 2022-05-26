const ccanvas = document.getElementById("myCanvas");
const cstatus = document.getElementById("status");


//const btn = document.getElementById("btnGenerate");
var ctx = ccanvas.getContext("2d");
ctx.font = "16px Arial";



window.onload = function() {

  ctx.font = "16px Arial";
}

if (ccanvas.getContext) {
    var ctx = ccanvas.getContext('2d');
    ctx.fillStyle = "rgba(200, 0, 0, 0.5)";
    ctx.fillRect(0, 0, 500, 500);

}


model = "";
(async () => {
  console.log("before load");
  model = await tf.loadLayersModel("model/model.json");
  console.log("after  load");
  cstatus.innerHTML = "GAN Model loaded";
}
)();

function genImage()
{

latent=tf.randomUniform([1,40],-0.95,1.0);
//latent=tf.randomNormal([1,40],0.5,0.6);


(async () => {
console.log("generating image");


ctx.fillText("Generating Mango...", 10, 50);


  product=model.predict(latent);

  product=product.toFloat().mul(127.5).add(127.5).asType("int32")

  product=product.squeeze();

  tf.browser.toPixels(product,ccanvas);

  latent.dispose();

})();




}

function animateK() {
  computing_animate_latent_space(model, 2, 20);

}

function image_enlarge(y, draw_multiplier) {
    if (draw_multiplier === 1) {
        return y;
    }
    let size = y.shape[0];
    return y.expandDims(2).tile([1, 1, draw_multiplier, 1]
    ).reshape([size, size * draw_multiplier, 3]
    ).expandDims(1).tile([1, draw_multiplier, 1, 1]
    ).reshape([size * draw_multiplier, size * draw_multiplier, 3])
}



async function computing_animate_latent_space(model, draw_multiplier, animate_frame) {
    const inputShape = model.inputs[0].shape.slice(1);
    const shift = tf.randomNormal(inputShape).expandDims(0);
    const freq = tf.randomNormal(inputShape, 0, .1).expandDims(0);

    //let c = document.getElementById("the_canvas");
    let i = 0;
    while (i < animate_frame) {
        i++;
        const y = tf.tidy(() => {
            const z = tf.sin(tf.scalar(i).mul(freq).add(shift));
            //const y = model.predict(z).squeeze().transpose([1, 2, 0]).div(tf.scalar(2)).add(tf.scalar(.5));
            const y = model.predict(z).squeeze().toFloat().mul(127.5).add(127.5).asType("int32")
            return image_enlarge(y, 1);
        });

        await tf.browser.toPixels(y, ccanvas);
        await tf.nextFrame();
    }
}
