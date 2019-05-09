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
let globalThreshold = 0.5;
let distThreshold = 30;
let resDict = {};

var detSlider = document.getElementById("detSensitivity");
var distSlider = document.getElementById("distSensitivity");

// Update the current slider value (each time you drag the slider handle)
detSlider.oninput = function() {
    globalThreshold = this.value / 100;
}

distSlider.oninput = function() {
    distThreshold = this.value;
}

function setup() {
    // Create a featureExtractor that can extract the already learned features from MobileNet
    // featureExtractor = ml5.featureExtractor('MobileNet', feModelReady);
    //noCanvas();

    var canvas = createCanvas(640, 480);
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

    // Initialise results dictionary
    // resDict['Focused'] = 0;
    // resDict['Distracted'] = 0;
    // resDict['AFK'] = 0;

}

// function feModelReady() {
//     select('#festatus').html('FeatureExtractor Loaded')
// }

function poseModelReady() {
    // select('#posestatus').html('PoseNet Loaded')
}

function draw() {
    if (showVideo) {
        image(video, 0, 0, width, height);
        select('#videoContainer').show();

    } else {
        select('#videoContainer').hide();
    }

    // if (doClassify) {
    //     classify();
    // }

    checkForFace();
}

function checkForFace() {

    if (poses.length > 0) {

        let faceCount = 0;

        for (let i = 0; i < poses.length; i++) {
            let pose = poses[i].pose;
            distBetweenEyes = euclideanDistance(pose.leftEye, pose.rightEye)

            if ((pose.leftEar.confidence >= globalThreshold || pose.rightEar.confidence >= globalThreshold) &&
                pose.leftEye.confidence >= globalThreshold && pose.rightEye.confidence >= globalThreshold && pose.nose.confidence > globalThreshold) {

                if (distBetweenEyes < distThreshold) {
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

// Add the current frame from the video to the classifier
function addExample(label) {
    // Get the features of the input video
    const features = featureExtractor.infer(video);
    // You can also pass in an optional endpoint, defaut to 'conv_preds'
    // const features = featureExtractor.infer(video, 'conv_preds');
    // You can list all the endpoints by calling the following function
    // console.log('All endpoints: ', featureExtractor.mobilenet.endpoints)

    // Add an example with a label to the classifier
    knnClassifier.addExample(features, label);
    updateCounts();
}

// Predict the current frame.
// function classify() {
//     // Get the total number of labels from knnClassifier
//     const numLabels = knnClassifier.getNumLabels();
//     if (numLabels <= 0) {
//         console.error('There is no examples in any label');
//         return;
//     }
//     // Get the features of the input video
//     const features = featureExtractor.infer(video);

//     // Use knnClassifier to classify which label do these features belong to
//     // You can pass in a callback function `gotResults` to knnClassifier.classify function
//     knnClassifier.classify(features, gotResults);
//     // You can also pass in an optional K value, K default to 3
//     // knnClassifier.classify(features, 3, gotResults);

//     // You can also use the following async/await function to call knnClassifier.classify
//     // Remember to add `async` before `function predictClass()`
//     // const res = await knnClassifier.classify(features);
//     // gotResults(null, res);
// }

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

    // // // When the A button is pressed, add the current frame
    // // from the video with a label of "Focused" to the classifier
    // buttonA = select('#addClassFocused');
    // buttonA.mousePressed(function() {
    //     addExample('Focused');
    // });

    // // When the B button is pressed, add the current frame
    // // from the video with a label of "Distracted" to the classifier
    // buttonB = select('#addClassDistracted');
    // buttonB.mousePressed(function() {
    //     addExample('Distracted');
    // });

    // // Reset buttons
    // resetBtnA = select('#resetFocused');
    // resetBtnA.mousePressed(function() {
    //     clearLabel('Focused');
    // });

    // resetBtnB = select('#resetDistracted');
    // resetBtnB.mousePressed(function() {
    //     clearLabel('Distracted');
    // });

    // // Predict button
    // buttonPredict = select('#buttonPredict');
    // buttonPredict.mousePressed(function() {
    //     doClassify = true;
    //     startTimer();
    // });

    // // Stop Predicting button
    // buttonPause = select('#buttonPause');
    // buttonPause.mousePressed(function() {
    //     doClassify = false;
    //     pauseTimer();
    // });

    // // Clear all classes button
    // buttonClearAll = select('#clearAll');
    // buttonClearAll.mousePressed(clearAllLabels);

    // // Load saved classifier dataset
    // buttonSetData = select('#load');
    // buttonSetData.mousePressed(loadMyKNN);

    // // Get classifier dataset
    // buttonGetData = select('#save');
    // buttonGetData.mousePressed(saveMyKNN);
}

// // Show the results
// function gotResults(err, result) {
//     // Display any error
//     if (err) {
//         console.error(err);
//     }

//     if (result.confidencesByLabel) {
//         const confidences = result.confidencesByLabel;

//         // result.label is the label that has the highest confidence
//         if (result.label) {
//             select('#result').html(result.label);
//             select('#confidence').html(`${confidences[result.label] * 100} %`);
//         }

//         select('#confidenceFocused').html(`${confidences['Focused'] ? confidences['Focused'] * 100 : 0} %`);
//         select('#confidenceDistracted').html(`${confidences['Distracted'] ? confidences['Distracted'] * 100 : 0} %`);

//         resDict['Focused'] += confidences['Focused'];
//         resDict['Distracted'] += confidences['Distracted'];
//         resDict['Total'] = resDict['Focused'] + resDict['Distracted'];
//         select('#progressBarFocused').attribute('style', `width:${ (resDict['Focused'] / resDict['Total']) * 100 }%`);
//         select('#progressBarDistracted').attribute('style', `width:${ (resDict['Distracted'] / resDict['Total']) * 100 }%`);

//         console.log('Focused: ' + (resDict['Focused'] / resDict['Total']) * 100);
//         console.log('Distracted: ' + (resDict['Distracted'] / resDict['Total']) * 100);

//     }

//     //classify();
// }

// // Update the example count for each label	
// function updateCounts() {
//     const counts = knnClassifier.getCountByLabel();

//     select('#exampleFocused').html(counts['Focused'] || 0);
//     select('#exampleDistracted').html(counts['Distracted'] || 0);
// }

// // Clear the examples in one label
// function clearLabel(label) {
//     knnClassifier.clearLabel(label);
//     updateCounts();
// }

// // Clear all the examples in all labels
// function clearAllLabels() {
//     knnClassifier.clearAllLabels();
//     updateCounts();
// }

// // Save dataset as myKNNDataset.json
// function saveMyKNN() {
//     knnClassifier.save();
// }

// // Load dataset to the classifier
// function loadMyKNN() {
//     knnClassifier.load('./myKNN.json', updateCounts);
// }

function euclideanDistance(p, p2) {
    xdiff = Math.pow((p.x - p2.x), 2);
    ydiff = Math.pow((p.y - p2.y), 2);
    return Math.sqrt(xdiff + ydiff)
}

// Timer based on https://medium.com/@olinations/an-accurate-vanilla-js-stopwatch-script-56ceb5c6f45b

// var startTimerButton = document.querySelector('.startTimer');
// var pauseTimerButton = document.querySelector('.pauseTimer');
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
        tInterval = setInterval(getShowTime, 1);
        // change 1 to 1000 above to run script every second instead of every millisecond. one other change will be needed in the getShowTime() function below for this to work. see comment there.   

        paused = 0;
        running = 1;
        timerDisplay.style.background = "#FF0000";
        timerDisplay.style.cursor = "auto";
        timerDisplay.style.color = "yellow";
        // startTimerButton.classList.add('lighter');
        // pauseTimerButton.classList.remove('lighter');
        // startTimerButton.style.cursor = "auto";
        // pauseTimerButton.style.cursor = "pointer";
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
        // startTimerButton.classList.remove('lighter');
        // pauseTimerButton.classList.add('lighter');
        // startTimerButton.style.cursor = "pointer";
        // pauseTimerButton.style.cursor = "auto";
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
    // startTimerButton.classList.remove('lighter');
    // pauseTimerButton.classList.remove('lighter');
    // startTimerButton.style.cursor = "pointer";
    // pauseTimerButton.style.cursor = "auto";
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
    var milliseconds = Math.floor((difference % (1000 * 60)) / 100);
    hours = (hours < 10) ? "0" + hours : hours;
    minutes = (minutes < 10) ? "0" + minutes : minutes;
    seconds = (seconds < 10) ? "0" + seconds : seconds;
    milliseconds = (milliseconds < 100) ? (milliseconds < 10) ? "00" + milliseconds : "0" + milliseconds : milliseconds;
    timerDisplay.innerHTML = hours + ':' + minutes + ':' + seconds + ':' + milliseconds;
}