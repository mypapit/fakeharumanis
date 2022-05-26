const ccanvas = document.getElementById("myCanvas");
const cstatus = document.getElementById("status");

var va = document.getElementById("va");
var vb = document.getElementById("vb");
var vc = document.getElementById("vc");
var vd = document.getElementById("vd");
var ve = document.getElementById("ve");
var vf = document.getElementById("vf");
var vg = document.getElementById("vg");
var vh = document.getElementById("vh");
var vi = document.getElementById("vi");
var vj = document.getElementById("vj");
var vk = document.getElementById("vk");
var vl = document.getElementById("vl");
var vm = document.getElementById("vm");
var vn = document.getElementById("vn");
var vo = document.getElementById("vo");
var vp = document.getElementById("vp");
var vq = document.getElementById("vq");
var vr = document.getElementById("vr");
var vs = document.getElementById("vs");
var vt = document.getElementById("vt");
var vu = document.getElementById("vu");
var vv = document.getElementById("vv");
var vw = document.getElementById("vw");
var vx = document.getElementById("vx");
var vy = document.getElementById("vy");
var vz = document.getElementById("vz");

var vaa = document.getElementById("vaa");
var vab = document.getElementById("vab");
var vac = document.getElementById("vac");
var vad = document.getElementById("vad");
var vae = document.getElementById("vae");
var vaf = document.getElementById("vaf");
var vag = document.getElementById("vag");
var vah = document.getElementById("vah");






//const btn = document.getElementById("btnGenerate");
var ctx = ccanvas.getContext("2d");
ctx.font = "16px Arial";


a = 0.1;
b = 0.2;
c = 0.3;
d = 0.4;
e = 0.5;
f = 0.6;
g = 0.7;
h = 0.8;
i = 0.9;
j = 0.0;



k = 0.1;
l = 0.2;
m = 0.3;
n = 0.4;
o = 0.5;
p = 0.6;
q = 0.7;
r = 0.8;
s = 0.9;
t = 0.0;
u = 0.7;
v = 0.8;
w = 0.9;
x = 0.0;
y = 0.6;
z = 0.7;

aa = 0.1;
ab = 0.2;
ac = 0.3;
ad = 0.4;
ae = 0.5;
af = 0.6;
ag = 0.7;
ah = 0.8;


var vtensor = [
  a,b,c,d,e,f,g,h,i,g,
  h,i,j,k,l,m,n,o,p,q,
  r,s,t,u,v,w,x,y,z,aa,
  ab,ac,ad,ae,af,ag,ah,ab,ac,ad
];


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






  a = parseFloat(va.value);
  b = parseFloat(vb.value);
  c = parseFloat(vc.value);
  d = parseFloat(vd.value);
  e = parseFloat(ve.value);
  f = parseFloat(vf.value);
  g = parseFloat(vg.value);
  h = parseFloat(vh.value);
  i = parseFloat(vi.value);
  j = parseFloat(vj.value);
  k = parseFloat(vk.value);
  l = parseFloat(vl.value);
  m = parseFloat(vm.value);
  n = parseFloat(vn.value);
  o = parseFloat(vo.value);
  p = parseFloat(vp.value);
  q = parseFloat(vq.value);
  r = parseFloat(vr.value);
  s = parseFloat(vs.value);
  t = parseFloat(vt.value);
  u = parseFloat(vu.value);
  v = parseFloat(vv.value);
  w = parseFloat(vw.value);
  x = parseFloat(vx.value);
  y = parseFloat(vy.value);
  z = parseFloat(vz.value);
  aa = parseFloat(vaa.value);
  ab= parseFloat(vab.value);
  ac = parseFloat(vac.value);
  ad = parseFloat(vad.value);
  ae = parseFloat(vae.value);
  af = parseFloat(vaf.value);
  ag = parseFloat(vag.value);
  ah = parseFloat(vah.value);


  var vtensor = [
    a,b,c,d,e,f,g,h,i,j,
    k,l,m,n,o,p,q,r,s,t,
    u,v,w,x,y,z,aa,ab,ac,
    ad,ae,af,ag,ah,ab,ac,ad,ae,af,ag
  ];

  ktensor = tf.tensor(vtensor);


 ktensor=ktensor.reshape([1,40]);

 ktensor.print();

//latent=tf.randomUniform(vtensor);


(async () => {
console.log("generating image");


ctx.fillText("Generating Mango...", 10, 50);


  product=model.predict(ktensor);

  product=product.toFloat().mul(127.5).add(127.5)

  product = product.asType("int32")


  product=product.squeeze();

  tf.browser.toPixels(product,ccanvas);

})();




}
