// Based on https://rosettacode.org/wiki/Hough_transform#Java

extern crate nalgebra as na;
extern crate image;

use std::f32;
use std::fs::File;
use std::path::Path;

use na::{DMatrix, min};

use image::{ImageLuma8, ImageBuffer, GenericImage, Pixel};

fn init(input_path: &str, output_path: &str, theta_axis_size: u32, r_axis_size: u32) {
    let img = image::open(&Path::new(input_path)).unwrap();

    let img_width = img.dimensions().0;
    let img_height = img.dimensions().1;

    let max_radius = ((img_width as f32).hypot(img_height as f32)).ceil();
    let r_axis_half = ((r_axis_size as f32) / 2.0).round();

    let min_contrast = 64;

    // getArrayDataFromImage returns an array of img_width x img_height,
    // with "average rgb value" a bit adjusted using
    // rgbValue = (int)(((rgbValue & 0xFF0000) >>> 16) * 0.30 + ((rgbValue & 0xFF00) >>> 8) * 0.59 + (rgbValue & 0xFF) * 0.11);
    let mut output_data = DMatrix::from_element(theta_axis_size as usize, r_axis_size as usize, 0);

    for (x, y, pixel) in img.pixels() {
        let r = pixel.channels()[0];
        let g = pixel.channels()[1];
        let b = pixel.channels()[2];

        let average_rgb = (((r as f32) * 0.3 + (g as f32) * 0.59 + (b as f32) * 0.11) / 3.0).round() as u32;

        // veeery simple edge detection
        if average_rgb > min_contrast {
            continue;
        }

        // we found an edge!
        for theta in (0..theta_axis_size).rev() {
            let r = (x as f32) * ((theta as f32).to_radians()).cos() + (y as f32) * ((theta as f32).to_radians()).sin();
            let r_scaled = ((r * r_axis_half / max_radius).round() + r_axis_half) as u32;
            // int rScaled = (int)Math.round(r * halfRAxisSize / maxRadius) + halfRAxisSize;
            // println!("r {} of {} scaled {}", r, theta, r_scaled);
            // let r = (x as f32) * (theta as f32).cos() + (y as f32) * (theta as f32).sin();

            // accumulate
            output_data[(theta as usize, r_scaled as usize)] += 1;
        }

        //for (int theta = thetaAxisSize - 1; theta >= 0; theta--)

        //println!("average: {}", average_rgb);

//        if !is_edge(x, y, img_width, img_height, min_contrast, average_rgb) {
//            continue;
//        }
    }


    // now write output image based on accumulator
    // @TODO writeOutputImage
    let accu_clone = output_data.clone().into_vector();
    let max = *accu_clone.iter().max().unwrap();// max what??

    let mut out = ImageBuffer::new(img_width, img_height); //ImageBuffer::new(img_width, img_height);

    for y in 0..img_height {
        for x in 0..img_width {

            let n = min(((output_data[(x as usize, y as usize)] as f32) * 255.0 / (max as f32)).round() as u32, 255) as u8;
            let pixel = image::Rgb([n, n, n]);
            out[(x, y)] = pixel;
        }
    }

    let ref mut fout = File::create(&Path::new("fractal.png")).unwrap();
    let _ = image::ImageRgb8(out).save(fout, image::PNG);
}

// fn is_edge(x: u32, y: u32, img_width: u32, img_height: u32, min_contrast: u32, average_rgb: u32) -> bool {
//     for i in (0..8).rev() {
//         if i == 4 {
//           continue;
//         }

//         let newx = x as i32 + (i as i32 % 3) - 1;
//         let newy = y as i32 + (i as i32/ 3) - 1;

//         if (newx < 0) || (newx >= img_width as i32) || (newy < 0) || (newy >= img_height as i32) {
//             continue;
//         }

//         if (average_rgb - center_value).abs() >= min_contrast {
//             return true;
//         }
//     }

//     false
// }

fn main() {
    init("/Users/max/Desktop/hough5.png", "/tmp/out.png", 640, 480);
}
