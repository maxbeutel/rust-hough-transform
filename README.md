# About

This is an implementation of the [Hough Transform algorithm](https://en.wikipedia.org/wiki/Hough_transform) for detecting lines in a greyscale image. The algorithm is implemented in Rust.

Unlike most examples I found on the web this implementation not only *dumps a visualization of the accumulator* (the so called "Hough Space") but also *shows the detected lines*, transformed back from the polar system to the original image.


# Installation

This project requires a working Rust compiler. The easiest way to get Rust installed is probably using [Rustup](https://www.rustup.rs/).

```
$ git clone git@github.com:maxbeutel/rust-hough-transform.git
$ cd rust-hough-transform
$ cargo build --release
```


# Usage

```
$ hough-transform <input-image> <output-houghspace> <output-lines> <theta-axis-scale-factor> <rho-axis-scale-factor> <accumulator-threshold>
```

## Arguments

* `<input-image>` - Path to the image which will be searched for lines. The algorithm assumes a greyscale image, black lines are considered edges.
* `<output-houghspace>` - Output path for a visualization of the Hough Space. Useful for debugging.
* `<output-lines>` - Output path for a visualization of the detected lines.
* `<theta-axis-scale-factor>` - Theta axis, scale factor, can improve accuracy, should be integer >= 1. See examples.
* `<rho-axis-scale-factor>` - Rho axis scale factor, should be integer >= 1. See examples.

## Examples


### Detecting square

![Square](https://raw.githubusercontent.com/maxbeutel/rust-hough-transform/master/data/sample-square.png "Square")
![Square result](https://raw.githubusercontent.com/maxbeutel/rust-hough-transform/master/data/lines-square.png "Square result")


```
$ target/release/hough-transform data/sample-square.png houghspace.png visualization.png 1 8 20

# max accumulator value: 20
# detected lines: 4
````

### Detecting square, 45 degree rotation

![Square, 45 degree rotation](https://raw.githubusercontent.com/maxbeutel/rust-hough-transform/master/data/sample-square-rotated45.png "Square, 45 degree rotation")
![Square, 45 degree rotation, result](https://raw.githubusercontent.com/maxbeutel/rust-hough-transform/master/data/lines-square-rotated45.png "Square, 45 degree rotation, result")

```
$ target/release/hough-transform data/sample-square-rotated45.png houghspace.png visualization.png 1 8 14

# max accumulator value: 14
# detected lines: 4
```

### Rectangle

![Rectangle](https://raw.githubusercontent.com/maxbeutel/rust-hough-transform/master/data/sample-rectangle.png "Rectangle")
![Rectangle result](https://raw.githubusercontent.com/maxbeutel/rust-hough-transform/master/data/lines-rectangle.png "Rectangle result")

```
$ target/release/hough-transform data/sample-rectangle.png houghspace.png visualization.png 1 8 300

# max accumulator value: 518
# detected lines: 44
```

### Pentagon

![Pentagon](https://raw.githubusercontent.com/maxbeutel/rust-hough-transform/master/data/sample-pentagon.png "Pentagon")
![Pentagon result](https://raw.githubusercontent.com/maxbeutel/rust-hough-transform/master/data/lines-pentagon.png "Pentagon result")

```
$ target/release/hough-transform data/sample-pentagon.png houghspace.png visualization.png 16 16 15

# max accumulator value: 27
# detected lines: 105
```

(Image from [rosettacode.org](https://rosettacode.org/wiki/Hough_transform))


# Possible improvements

* The edge detection is very unsophisticated, as can be observed in the "Rectangle" example, also detection of lines in the "Pentagon" example would benefit from a more accurate edge detection algorithm. The current edge detection only checks of the pixel has an RGB value of 0/0/0 (black).
* Pre-compute lookup tables for sinus/cosinus for all angles to increase performance and reduce costly floating point arithmetic.
* More test cases could be added, currently some methods take an `Image` as argument, which makes them hard to test.
* The style of this project could be a lot more functional.
* Add travis build to stay up to date with new Rust versions and updated libraries.
* The threshold for filtering the Hough Space could be defined automatically, like 70% of the maximum value in the accumulator. Filtering can also be improved, e. g. by finding local maxima.
