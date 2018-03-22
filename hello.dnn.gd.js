var nj = require('numjs');

var inputs = nj.array([
    [0, 0, 1],  // -> 0
    [1, 1, 1],  // -> 0
    [1, 0, 1],  // -> 1
    [0, 1, 1]   // -> 1
]);

var test_result = nj.array([[0, 0, 1, 1]]).T;

// Gradient slope : should neither too small nor too big
// Needed to be try with different values for better result
var alpha = 10;

// Size of hidden layers
var hidden = 32;

var weights_zero = nj.array(rand(3, hidden));
var weights_one = nj.array(rand(hidden, 1));

function train(inputs, test_result, iterations) {
    for(var i = 0; i < iterations; i++) {
        var layer_zero = inputs;

        var layer_one = nj.sigmoid( layer_zero.dot(weights_zero) );
        var layer_two = nj.sigmoid( layer_one.dot(weights_one) );

        var layer_two_error = layer_two.subtract(test_result);

        if ((i % 10000) == 0) {
            console.log(i + " - Error: " + nj.mean(nj.abs(layer_two_error)));
        }

        // Backpropagation (sending back layer_two errors to layer_one)
        var layer_two_delta = layer_two_error.multiply( curve(layer_two) );
        var layer_one_error = layer_two_delta.dot( weights_one.T );
        var layer_one_delta = layer_one_error.multiply( curve(layer_one) );

        // Adjusting weights
        weights_one = weights_one.subtract(
            layer_one.T.dot(layer_two_delta).multiply(alpha)
        );

        weights_zero = weights_zero.subtract(
            layer_zero.T.dot(layer_one_delta).multiply(alpha)
        );
    }
}

function think(inputs) {
    var layer_one = nj.sigmoid( inputs.dot(weights_zero) );
    var layer_two = nj.sigmoid( layer_one.dot(weights_one) );

    return layer_two;
}

/* === Some math functions === */

// NumJs come with sigmoid function.
// But no sigmoid derivative. This is it.
function curve(nums) {
    nums = nums.tolist();
    var result = [];
    for(var i = 0; i < nums.length; i++) {
        result[i] = [];
        for(var ii=0; ii<nums[i].length; ii++) {
            result[i][ii] = nums[i][ii] * (1 - nums[i][ii]);
        }
    }

    return nj.array(result);
}

// Random number between 1 and -1
// return as 2D array
function rand(rows, cols) {
    var result = [];
    for(var i=0; i<rows; i++) {
        result[i] = [];
        for(var ii=0; ii<cols; ii++) {
            result[i][ii] = 2 * Math.random() - 1;
        }
    }

    return result;
}

/* === Training === */
train(inputs, test_result, 60000);

/* === Testing === */
var test_data = [
    [1, 0, 0],
    [1, 1, 0]
];

console.log( think( nj.array(test_data) ) );
