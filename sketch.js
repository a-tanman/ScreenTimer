// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
KNN Classification on Webcam Images with mobileNet. Built with p5.js
=== */
let video;
// Create a KNN classifier
const knnClassifier = ml5.KNNClassifier();
let featureExtractor;
let poseNet;
let poses = [];
let showDots = true;
let showVideo = true;
let doClassify = false;
let globalThreshold = 0.3;
let distThreshold = 0.075;
let resDict = {};

var detSlider = document.getElementById("detSensitivity");
var distSlider = document.getElementById("distSensitivity");

// Update the current slider value (each time you drag the slider handle)
detSlider.oninput = function() {
    globalThreshold = 0.8 - this.value / 8;
}

distSlider.oninput = function() {
    distThreshold = 0.15 - this.value * 0.0015;
}

function setup() {

    let heightAndWidth = calculateWidthAndHeight(displayWidth);

    var canvas = createCanvas(heightAndWidth.w, heightAndWidth.h);
    video = createCapture(VIDEO);
    video.size(width, height);

    //Create a new poseNet method with a single detection
    poseNet = ml5.poseNet(video, poseModelReady);

    frameRate(20);
    // This sets up an event that fills the global variable "poses"
    // with an array every time new poses are detected
    poseNet.on('pose', function(results) {
        poses = results;
        // console.log(poses);
    });

    // Append it to the videoContainer DOM element
    video.parent('videoContainer');
    canvas.parent('videoContainer');
    video.hide();
    // Create the UI buttons
    createButtons();

}

function poseModelReady() {
    console.log('Model Loaded');
    // select('#posestatus').html('PoseNet Loaded')
}

function draw() {

    if (showVideo) {
        image(video, 0, 0, width, height);
        select('#videoContainer').show();

    } else {
        select('#videoContainer').hide();
    }

    checkForFace();
}

function calculateWidthAndHeight(windowWidth) {
    width = min(640, windowWidth * 0.9)
    return {
        w: width,
        h: int(width * 3 / 4)
    };
}

function checkForFace() {

    if (poses.length > 0) {

        let faceCount = 0;

        for (let i = 0; i < poses.length; i++) {
            let pose = poses[i].pose;
            distBetweenEyes = euclideanDistance(pose.leftEye, pose.rightEye)
            distBetweenEyesProportion = distBetweenEyes / width;


            if ((pose.leftEar.confidence >= globalThreshold || pose.rightEar.confidence >= globalThreshold) &&
                pose.leftEye.confidence >= globalThreshold && pose.rightEye.confidence >= globalThreshold && pose.nose.confidence > globalThreshold) {

                if (distBetweenEyesProportion < distThreshold) {
                    select('#facestatus').html('Face not detected / not looking at screen / too far away?');
                    select('#facestatus').style('color', '#990000');
                    pauseTimer();
                } else {
                    select('#facestatus').html('Face detected!');
                    select('#facestatus').style('color', '#4CAF50');

                    startTimer();
                    faceCount++;
                }
            }

            if (showDots == true) {
                for (let j = 0; j < pose.keypoints.length; j++) {
                    // A keypoint is an object describing a body part (like rightArm or leftShoulder)
                    let keypoint = pose.keypoints[j];
                    // Only draw an ellipse is the pose probability is bigger than globalThreshold
                    if (keypoint.score >= globalThreshold && faceCount > 0) {
                        fill(255, 0, 0);
                        noStroke();
                        ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
                    }
                }
            }
        }

        if (faceCount == 0) {
            select('#facestatus').html('Face not detected / not looking at screen / too far away?');
            select('#facestatus').style('color', '#990000');
            // console.log(poses)
            pauseTimer();
        }

    }
}

// A util function to create UI buttons
function createButtons() {

    cbRedDots = select('#redDots');
    cbRedDots.mouseClicked(function() {
        if (cbRedDots.checked()) {
            showDots = true;
        } else {
            showDots = false;
        }
    });

    cbVideo = select('#video');
    cbVideo.mouseClicked(function() {
        if (cbVideo.checked()) {
            showVideo = true;
        } else {
            showVideo = false;
        }
    });
}

function euclideanDistance(p, p2) {
    xdiff = Math.pow((p.x - p2.x), 2);
    ydiff = Math.pow((p.y - p2.y), 2);
    return Math.sqrt(xdiff + ydiff)
}

// Timer based on https://medium.com/@olinations/an-accurate-vanilla-js-stopwatch-script-56ceb5c6f45b

var timerDisplay = document.querySelector('.timer');
var startTime;
var updatedTime;
var difference;
var tInterval;
var savedTime;
var paused = 0;
var running = 0;

function startTimer() {
    if (!running) {
        startTime = new Date().getTime();
        tInterval = setInterval(getShowTime, 1000);
        // change 1 to 1000 above to run script every second instead of every millisecond. one other change will be needed in the getShowTime() function below for this to work. see comment there.   

        paused = 0;
        running = 1;
        timerDisplay.style.background = "#FF0000";
        timerDisplay.style.cursor = "auto";
        timerDisplay.style.color = "yellow";
    }
}

function pauseTimer() {
    if (!difference) {
        // if timer never started, don't allow pause button to do anything
    } else if (!paused) {
        clearInterval(tInterval);
        savedTime = difference;
        paused = 1;
        running = 0;
        timerDisplay.style.background = "#A90000";
        timerDisplay.style.color = "#690000";
        timerDisplay.style.cursor = "pointer";
    } else {
        // if the timer was already paused, when they click pause again, start the timer again
        // Changed to NOT start timer again
        //startTimer();
    }
}

function resetTimer() {
    clearInterval(tInterval);
    savedTime = 0;
    difference = 0;
    paused = 0;
    running = 0;
    timerDisplay.innerHTML = 'Start Timer!';
    timerDisplay.style.background = "#A90000";
    timerDisplay.style.color = "#fff";
    timerDisplay.style.cursor = "pointer";
}

function getShowTime() {
    updatedTime = new Date().getTime();
    if (savedTime) {
        difference = (updatedTime - startTime) + savedTime;
    } else {
        difference = updatedTime - startTime;
    }
    // var days = Math.floor(difference / (1000 * 60 * 60 * 24));
    var hours = Math.floor((difference % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    var minutes = Math.floor((difference % (1000 * 60 * 60)) / (1000 * 60));
    var seconds = Math.floor((difference % (1000 * 60)) / 1000);
    // var milliseconds = Math.floor((difference % (1000 * 60)) / 100);
    hours = (hours < 10) ? "0" + hours : hours;
    minutes = (minutes < 10) ? "0" + minutes : minutes;
    seconds = (seconds < 10) ? "0" + seconds : seconds;
    // milliseconds = (milliseconds < 100) ? (milliseconds < 10) ? "00" + milliseconds : "0" + milliseconds : milliseconds;
    timerDisplay.innerHTML = hours + ':' + minutes + ':' + seconds // + ':' + milliseconds;
}