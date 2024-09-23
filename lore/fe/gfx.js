import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
//const controls = new OrbitControls( camera, renderer.domElement );

var scene, camera, renderer;
var controls;
var mywidth, myheight, mydepth;
var instances;

var innercircles = 300;
var outercircles = 4;
var inner_radius = 10.0;
var outer_radius = 100.0;
var sizing = 500.0;
let spacing = 4.0;

let onecircles = [];
let outercirc = [];
function circleup()
{
    var ang, innersize = inner_radius * innercircles;
    var onecircle;
    onecircles=[];
    for( var i=0; i<innercircles; i++ ) {
        onecircle = [];
        for( var j=0; j<outercircles; j++ ) {
            var x,y;
            ang = 2*Math.PI*(i/innercircles) + 0.1*2*Math.PI*(j/outercircles);
            x = innersize * Math.cos( ang );
            y = innersize * Math.sin( ang );
            onecircle.push( [x,y] );
        }
        onecircles.push(onecircle);
    }
    outercirc=[];
    for( var i=0; i<outercircles; i++ ) {
        ang = 2*Math.PI*(i/outercircles);
        outercirc.push( [outer_radius * Math.cos( ang ), outer_radius * Math.sin( ang )] );
    }
}

export function threeCanvas(el)
{
    var i, j, k;

    circleup();

    scene = new THREE.Scene();
    scene.background = new THREE.Color( 0x000000 );

    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000000 );
    renderer = new THREE.WebGLRenderer({
        antialias: true
        //alpha: true
    });
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.shadowMap.enabled = true;
    renderer.setSize( window.innerWidth-16, window.innerHeight-16 );
    var canvas;
    if( el.children.length > 0 )
        el.insertBefore( canvas=renderer.domElement, el.children[0] );
    else
        el.appendChild( canvas=renderer.domElement );
   
    controls = new OrbitControls( camera, renderer.domElement );

    var light1 = new THREE.PointLight(0x0000ff, 0.5, 0);
    light1.position.set(0.5*mywidth, 0.5*myheight, 0.5*mydepth);
    scene.add(light1);
    var light2 = new THREE.PointLight(0xff0000, 0.5, 0);
    light2.position.set(0.5*mywidth, 0.5*myheight, 0.5*mydepth);
    scene.add(light2);

    var light3 = new THREE.AmbientLight(0xffffff, 1.0);
    scene.add(light3);
    
    const geometry = new THREE.SphereGeometry( sizing, 6, 6 )
    const material = new THREE.MeshLambertMaterial( {color: 0xffffff} );
    //material.opacity = 0.33;
    //material.transparent = true;
    //material.wireframe = true;
    instances = new THREE.InstancedMesh( geometry, material, 16*innercircles*outercircles );
    instances.instanceMatrix.setUsage( THREE.DynamicDrawUsage ); // will be updated every frame

    var lines = 16;
    var oneframe = innercircles*outercircles;
    for( k=0; k<lines; k++ ) {
        for( i=0; i<innercircles; i++ ) {
            for( j=0; j<outercircles; j++ ) {
                var r = 0.5 + j/outercircles;// + (i/mydepth)*0.5;
                var g = 0.5;
                var b = 0.5 + (outercircles-j)/outercircles;// + (i/mydepth)*0.5;
                instances.setColorAt( k*oneframe + i*outercircles + j, new THREE.Color( r,g,b ) );
                instances.setMatrixAt( k*oneframe + i*outercircles + j, new THREE.Matrix4().makeTranslation( 0, 0, -100000 ) );
            }
        }
    }
    instances.instanceColor.needsUpdate = true;
    //instances.castShadow = instances.receiveShadow = true;
    scene.add(instances);

    //camera.position.set( mywidth*spacing*0.5, 0, -mydepth*spacing*0.5 );
    camera.position.set( 3400, 8400, -10300 );
    camera.lookAt(new THREE.Vector3(0, 0, 0) );

    controls.target = new THREE.Vector3(0, 0, 0);
    controls.update();
}

var zvals=[], yvals=[];
export function threeRender()
{
    var cube;

    var i,j,k;
    var n=0;
    var dtn = new Date().getTime();

    var x, y, z, ang;
    var ix, iy;
    if( zvals.length == 0 ) {
        for( j=0; j<outercircles; j++ ) {
            zvals.push( outer_radius * outercircles * Math.cos( 2*Math.PI*(j/(1.33*outercircles)) ) );
            yvals.push( outer_radius * outercircles * Math.sin( 2*Math.PI*(j/(1.33*outercircles)) ) );
        }
        console.log(zvals);
    }

    controls.update();

    var lines = distortion[0].length;
    var oneframe = innercircles*outercircles;
    for( k=0; k<lines; k++ ) {
        for( i=0; i<innercircles; i++ ) {
            for( j=0; j<outercircles; j++ ) {
                //ang = 2*Math.PI*(j/(1.33*outercircles));
                if( i == innercircles-1 ) {
                    z = zvals[j] * ( i/innercircles );
                } else {
                    z = zvals[j] * ( i/innercircles ) + zvals[j+1] * ( (innercircles-i)/innercircles );//outer_radius * outercircles * Math.cos( ang );
                }
                
                ix = onecircles[i][j][0]*(j<distortion.length?1+distortion[j][k]:1);
                iy = onecircles[i][j][1]*(j<distortion.length?1+distortion[j][k]:1) + yvals[j];//outer_radius * outercircles * Math.sin( ang );
                
                instances.setMatrixAt( k*oneframe + i*outercircles + j, new THREE.Matrix4().makeTranslation( ix + k*spacing*inner_radius*innercircles, iy, z ) );
    //          instances.setColorAt( i*oneframe + j*mywidth + k, new THREE.Color( red/255.0, green/255.0, blue/255.0 ) );
            }
        }
    }
    instances.instanceMatrix.needsUpdate = true;
    //instances.instanceColor.needsUpdate = true;
    //instances.computeBoundingSphere();

    renderer.render( scene, camera );
}
