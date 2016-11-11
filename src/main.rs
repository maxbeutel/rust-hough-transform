// Based on https://rosettacode.org/wiki/Hough_transform#Java

extern crate nalgebra as na;
extern crate image;

use std::f32;
use std::fs::File;
use std::path::Path;

use na::{DMatrix, min};

use image::{ImageBuffer, GenericImage, Pixel};

fn init(
    input_path: &str,
    output_path: &str,
    min_contrast: u32,
    theta_axis_size: u32,
    r_axis_size: u32
) {
    let img = image::open(&Path::new(input_path)).unwrap();

    let img_width = img.dimensions().0;
    let img_height = img.dimensions().1;

    let max_radius = ((img_width as f32).hypot(img_height as f32)).ceil();
    let r_axis_half = ((r_axis_size as f32) / 2.0).round();

    let mut accumulator = DMatrix::from_element(theta_axis_size as usize, r_axis_size as usize, 0);

    // iterate over each pixel, if it's an edge, count it in the accumulator
    for (x, y, pixel) in img.pixels() {
        let r = pixel.channels()[0];
        let g = pixel.channels()[1];
        let b = pixel.channels()[2];

        let average_rgb = (((r as f32) * 0.3 + (g as f32) * 0.59 + (b as f32) * 0.11) / 3.0).round() as u32;

        if !is_edge(&accumulator, (x, y, average_rgb), (img_width, img_height), min_contrast) {
            continue;
        }

        // we found an edge, now go through all possible angles
        for theta in (0..theta_axis_size).rev() {
            let r = (x as f32) * ((theta as f32).to_radians()).cos() + (y as f32) * ((theta as f32).to_radians()).sin();
            let r_scaled = ((r * r_axis_half / max_radius).round() + r_axis_half) as u32;

            // accumulate
            accumulator[(theta as usize, r_scaled as usize)] += 1;
        }
    }

    // now write output image based on accumulator
    let accu_clone = accumulator.clone().into_vector();
    let max = *accu_clone.iter().max().unwrap(); // max what??

    let out_img_width = accumulator.nrows() as u32;
    let out_img_height = accumulator.ncols() as u32; // this seems kind of wrong, cols are the height???

    let mut out = ImageBuffer::new(out_img_width, out_img_height);

    for y in 0..out_img_height {
        for x in 0..out_img_width {
            let n = min(((accumulator[(x as usize, y as usize)] as f32) * 255.0 / (max as f32)).round() as u32, 255) as u8;
            let pixel = image::Rgb([n, n, n]);

            out[(x, y)] = pixel;
        }
    }

    let ref mut fout = File::create(&Path::new(output_path)).unwrap();
    let _ = image::ImageRgb8(out).save(fout, image::PNG);
}

fn is_edge(accumulator: &DMatrix<u32>, pixel_data: (u32, u32, u32), img_dimensions: (u32, u32), min_contrast: u32) -> bool {
    let average_rgb = pixel_data.2;

    // 3 x 3 matrix to check neighboring pixels
    for i in (0..8).rev() {
        if i == 4 {
            continue;
        }

        let new_x = pixel_data.0 as i32 + (i as i32 % 3) - 1;
        let new_y = pixel_data.1 as i32 + (i as i32/ 3) - 1;

        if (new_x < 0) || (new_x >= img_dimensions.0 as i32) || (new_y < 0) || (new_y >= img_dimensions.1 as i32) {
            continue;
        }

        let neighbor_average_rgb = accumulator[(new_x as usize, new_y as usize)];

        if ((neighbor_average_rgb as i32) - (average_rgb as i32)).abs() >= (min_contrast as i32) {
            return true;
        }
    }

    false
}

fn main() {
    let input_path = "/Users/max/Desktop/hough5.png";
    let output_path = "/tmp/hough_space.png";
    let min_contrast = 85;
    let theta_axis_size = 640;
    let r_axis_size = 480;

    init(input_path, output_path, min_contrast, theta_axis_size, r_axis_size);
}
