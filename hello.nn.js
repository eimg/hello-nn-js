var inputs = [
    [0, 0, 1],  // -> 0
    [1, 1, 1],  // -> 1
    [1, 0, 1],  // -> 1
    [0, 1, 1]   // -> 0
];

var test_result = [0, 1, 1, 0];

var weights = [ rand(), rand(), rand() ];

function train(inputs, test_result, iterations) {
    for(var i = 0; i < iterations; i++) {
        var output = think(inputs);
        var error = array_sub(test_result, output);

        var adjustment = dot(T(inputs), array_multi(error, sigmoid_derivative(output)));

        weights = array_add(weights, adjustment);
    }
}

function think(inputs) {
    return sigmoid(dot(inputs, weights));
}

/* === Some math functions === */

// Normalize given numbers between 0 and 1
function sigmoid(nums) {
    var result = [];
    for(var i = 0; i < nums.length; i++) {
        result[i] = 1 / (1 + Math.exp(-nums[i]));
    }

    return result;
}

// Check if numbers are within sigmoid curve
function sigmoid_derivative(nums) {
    var result = [];
    for(var i = 0; i < nums.length; i++) {
        result[i] = nums[i] * (1 - nums[i]);
    }

    return result;
}

// Matrix multiplication
function dot(a, b) {
    var result = [];
    for(var i=0; i<a.length; i++) {
        var cell = 0;
        for(var ii=0; ii<a[i].length; ii++) {
            cell += a[i][ii] * b[ii];
        }
        result[i] = cell;
    }

    return result;
}

// Substract two same-size arrays
function array_sub(a, b) {
    var result = [];
    for(var i=0; i<a.length; i++) {
        result[i] = a[i] - b[i];
    }

    return result;
}

// Add two same-size arrays
function array_add(a, b) {
    var result = [];
    for(var i=0; i<a.length; i++) {
        result[i] = a[i] + b[i];
    }

    return result;
}

// Multiply two same-size arrays
function array_multi(a, b) {
    var result = [];
    for(var i=0; i<a.length; i++) {
        result[i] = a[i] * b[i];
    }

    return result;
}

// Array matrix transpose (column to row)
function T(a){
    var result = [];
    for(var i = 0; i < a[0].length; i++){
        result.push([]);
    };

    for(var i = 0; i < a.length; i++){
        for(var j = 0; j < a[0].length; j++){
            result[j].push(a[i][j]);
        };
    };

    return result;
}

// Random number between 1 and -1
function rand() {
    return 2 * Math.random() - 1;
}

/* === Random Weights === */
console.log("Initial Weights: " + weights + "\n");

/* === Training === */
train(inputs, test_result, 10000);

/* === Trained Weight === */
console.log("Trained Weights: " + weights + "\n");

/* === Testing === */
console.log( think( [[1, 0, 0], [0, 1, 0]] ) );
