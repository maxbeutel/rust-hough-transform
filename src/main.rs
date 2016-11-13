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

    let mut greyscale_data = DMatrix::from_element(img_width as usize, img_height as usize, 0);

    // Flip y axis when reading image
    for y in 0..img_height {
        for x in 0..img_width {
            let pixel = img.get_pixel(x, y);
            let r = pixel.channels()[0];

            // @TODO convert to greyscale here!
            let y_offset = img_height - 1 - y;
            greyscale_data[(x as usize, y_offset as usize) ] = r;
            //println!("at {} {}/{} = {}", (y * img_width + x), x, y_offset, r);
        }
    }

    let max_radius = ((img_width as f32).hypot(img_height as f32)).ceil();
    let r_axis_half = ((r_axis_size as f32) / 2.0).round();
    let mut accumulator = DMatrix::from_element(theta_axis_size as usize, r_axis_size as usize, 0);

    for y in 0..img_height {
        for x in 0..img_width {
            let i = greyscale_data[(x as usize, y as usize)];
            //println!("color at {}/{} is {}", x, y, i);
            if is_edge(&greyscale_data, (x, y, i as u32), (img_width, img_height), min_contrast) {
                //println!("detected edge: {}/{}", x, y);

                // we found an edge, now go through all possible angles
                for theta in (0..theta_axis_size).rev() {
                    let theta_radians = theta as f64 * std::f64::consts::PI / theta_axis_size as f64; // deg * pi / 180
                    let sin = theta_radians.sin();
                    let cos = theta_radians.cos();

                    let r = (x as f64) * cos + (y as f64) * sin;
                    let r_scaled = ((r * r_axis_half as f64 / max_radius as f64).round() + r_axis_half as f64) as u32;

                    //println!("accu {}/{} at {}/{}", theta, r, x, y);

                    // accumulate
                    accumulator[(theta as usize, r_scaled as usize)] += 1;
                }
            }
        }
    }

    // now write output image based on accumulator
    let accu_clone = accumulator.clone().into_vector();
    let max_accumualtor_value = *accu_clone.iter().max().unwrap();

    let out_img_width = accumulator.nrows() as u32;
    let out_img_height = accumulator.ncols() as u32; // this seems kind of wrong, cols are the height???

    //println!("out buffered {}/{}", out_img_width, out_img_height);

    let mut out = ImageBuffer::new(out_img_width, out_img_height);

    for y in 0..out_img_height {
        for x in 0..out_img_width {
            let n = min(((accumulator[(x as usize, y as usize)] as f32) * 255.0 / (max_accumualtor_value as f32)).round() as u32, 255) as u8;
            //println!("n {} at {}/{}", n, x, y);
            let pixel = image::Rgb([n, n, n]);

            out[(x, out_img_height - y - 1)] = pixel;
        }
    }

    let ref mut fout = File::create(&Path::new(output_path)).unwrap();
    let _ = image::ImageRgb8(out).save(fout, image::PNG);
}

fn is_edge(greyscale_data: &DMatrix<u8>, pixel_data: (u32, u32, u32), img_dimensions: (u32, u32), min_contrast: u32) -> bool {
    let average_rgb = pixel_data.2;

    //println!("at {}/{} - center value: {}", pixel_data.0, pixel_data.1, average_rgb);

    // 3 x 3 matrix to check neighboring pixels
    // @TODO can use nalgebtra matrix for this?
    for i in (0..9).rev() {
        if i == 4 {
            continue;
        }

        let new_x = pixel_data.0 as i32 + (i as i32 % 3) - 1;
        let new_y = pixel_data.1 as i32 + (i as i32 / 3) - 1;

        //println!("new xy {}/{}", new_x, new_y);

        if (new_x < 0) || (new_x >= img_dimensions.0 as i32) || (new_y < 0) || (new_y >= img_dimensions.1 as i32) {
            continue;
        }

        let r = greyscale_data[(new_x as usize, new_y as usize)] as u32;
        // @TODO ^ fix this, this is not average RGB, need to convert to greyscale!

        //println!("neighbor rgb: {}", r);

        if ((r as i32) - (average_rgb as i32)).abs() >= (min_contrast as i32) {
            return true;
        }
    }

    false
}

fn main() {
    let input_path = "/Users/max/Desktop/hough5.png";
    let output_path = "hough_space.png";

    let theta_axis_size = 640;
    let r_axis_size = 480;
    let min_contrast = 64;

    init(input_path, output_path, min_contrast, theta_axis_size, r_axis_size);
}
