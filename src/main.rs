// Based on https://rosettacode.org/wiki/Hough_transform#Java

extern crate nalgebra as na;
extern crate image;

use std::io::Write;
use std::env;
use std::process;
use std::io;
use std::f32;
use std::fs::File;
use std::path::Path;
use std::str::FromStr;

use image::{ImageBuffer, GenericImage, Pixel};

#[inline(always)]
fn rgb_to_greyscale(r: u8, g: u8, b: u8) -> u8 {
    return ((r as f32 + g as f32 + b as f32) / 3.0).round() as u8;
}

// #[inline(always)]
// fn deg2rad(axis_size: u32, deg: f64) -> f64 {
//     // this is IMPORTANT: compute radians based on the theta axis size, which can be larger than 180 deg!
//     let radians = deg as f64 * std::f64::consts::PI / axis_size as f64;
//     return radians;
// }

fn init(
    input_img_path: &str,
    houghspace_img_path: &str,
    line_visualization_img_path: &str,
    rho_axis_scale_factor: u32,
    houghspace_filter_threshold: u32
) {
    let mut img = image::open(&Path::new(input_img_path)).unwrap();
    let (img_width, img_height) = img.dimensions();

    // @FIXME this code doesn't work for landscape and non-landscape images!
    let max_line_length = ((img_width as f32).hypot(img_height as f32)).ceil();

    // @FIXME when making this configurable, deg2rad function must be used
    // for angle calculations!
    // @FIXME this should actually be 180 and then add 1 at the places where needed
    let theta_axis_size = 180 + 1;

    // making rho axis size larger increases accuracy a lot (compare 8 * maxlen vs 2 * maxlen)
    // (preventing that different angles generate the same rho)
    let rho_axis_size = (max_line_length as u32) * rho_axis_scale_factor; // 16
    let rho_axis_half = ((rho_axis_size as f32) / 2.0).round();

    let mut accumulator: na::DMatrix<u32> = na::DMatrix::new_zeros(theta_axis_size as usize, rho_axis_size as usize);

    for y in 0..img_height {
        for x in 0..img_width {
            let pixel = img.get_pixel(x, y);
            let y_inverted = img_height - y - 1;

            let i = rgb_to_greyscale(pixel.channels()[0], pixel.channels()[1], pixel.channels()[2]);

            // found an edge
            if i < 1 {
                for theta in 1..theta_axis_size {
                    let sin = (theta as f64).to_radians().sin();//   deg2rad(theta_axis_size, theta as f64).sin();
                    let cos = (theta as f64).to_radians().cos();// deg2rad(theta_axis_size, theta as f64).cos();

                    let rho = (x as f64) * cos + (y_inverted as f64) * sin;
                    let rho_scaled = ((rho * rho_axis_half as f64 / max_line_length as f64).round() + rho_axis_half as f64) as u32;

                    accumulator[(theta as usize, rho_scaled as usize)] += 1;
                }
            }
        }
    }

    // now write output image based on accumulator
    let accu_clone = accumulator.clone().into_vector();
    let max_accumulator_value = *accu_clone.iter().max().unwrap();
    println!("max accu value: {}", max_accumulator_value);

    let out_img_width = accumulator.nrows() as u32;
    let out_img_height = accumulator.ncols() as u32;

    // dump hough space as image
    let mut out = ImageBuffer::new(out_img_width, out_img_height);

    for y in 0..out_img_height {
        for x in 0..out_img_width {
            let n = na::min(((accumulator[(x as usize, y as usize)] as f32) * 255.0 / (max_accumulator_value as f32)).round() as u32, 255) as u8;
            let pixel = image::Rgb([n, n, n]);

            out[(x, out_img_height - y - 1)] = pixel;
        }
    }

    let ref mut fout = File::create(&Path::new(houghspace_img_path)).unwrap();
    let _ = image::ImageRgb8(out).save(fout, image::PNG);

    // filter hough space and dump lines
    for theta in 0..theta_axis_size {
        for rho_scaled in 0..rho_axis_size {
            let val = accumulator[(theta as usize, rho_scaled as usize)];

            if val < houghspace_filter_threshold {
                continue;
            }

            // @TODO rename rho_original to rho
            let rho_original = (rho_scaled as f64 - rho_axis_half as f64) * max_line_length as f64 / rho_axis_half as f64;
            //println!("{} {} {}", theta, rho_scaled, rho_original);

            let (p1_x, p1_y, p2_x, p2_y) = transform_lines(
                theta as f64,
                rho_original,
                img_width,
                img_height
            );

            //println!("(transform) {} {} {}/{} to {}/{}", theta, rho_original, p1_x.round(), p1_y.round(), p2_x.round(), p2_y.round());

            let mut clipped_x1 = 0.0;
            let mut clipped_y1 = 0.0;
            let mut clipped_x2 = 0.0;
            let mut clipped_y2 = 0.0;

            liang_barsky(
                0.0, img_width as f64 - 1.0, 0.0, img_height as f64 - 1.0,
                p1_x, p1_y, p2_x, p2_y,
                &mut clipped_x1, &mut clipped_y1, &mut clipped_x2, &mut clipped_y2
            );

            //println!("(clip) {}/{} to {}/{}", clipped_x1.round(), clipped_y1.round(), clipped_x2.round(), clipped_y2.round());

            draw_line(
                &mut img,
                clipped_x1.round() as i32,
                img_height as i32 - 1 - clipped_y1.round() as i32,
                clipped_x2.round() as i32,
                img_height as i32 - 1 - clipped_y2.round() as i32,
                image::Rgba([255, 0, 0, 1])
            );
        }
    }

    let ref mut visualization_fout = File::create(&Path::new(line_visualization_img_path)).unwrap();
    let _ = img.save(visualization_fout, image::PNG);
}

fn transform_lines(
    theta: f64,
    rho: f64,
    img_width: u32,
    img_height: u32
) -> (f64, f64, f64, f64) {
    let mut p1_x = 0.0 as f64;
    let mut p1_y = 0.0 as f64;

    let mut p2_x = 0.0 as f64;
    let mut p2_y = 0.0 as f64;

    // @FIXME theta_reverse is a really stupid name
    // @FIXME move computation of theta_reverse/theta remaining to beginning of the function

    let theta = if theta > 180.0 { theta - 180.0 } else { theta };

    // special cases
    if theta == 0.0 || theta == 180.0 {
        p1_x = rho.abs(); // not tested
        p1_y = img_height as f64;

        p2_x = rho.abs(); // not tested
        p2_y = 0.0;
    } else if theta == 90.0 {
        p1_x = 0.0;
        p1_y = rho.abs(); // not tested

        p2_x = img_width as f64;
        p2_y = rho.abs(); // not tested
    } else if theta > 0.0 && theta < 90.0  {
        let theta_reverse = theta;
        let theta_remaining = 90.0 - theta_reverse;

        // start
        p1_x = 0.0;
        p1_y = rho.abs() / theta.to_radians().sin();//deg2rad(theta_axis_size, theta).sin();

        // end
        p2_x = rho.abs() / theta_remaining.to_radians().sin();//deg2rad(theta_axis_size, theta_remaining).sin();
        p2_y = 0.0;

        //println!("{} {} {} {} {}/{} {}/{}", theta, theta_reverse, theta_remaining, rho, p1_x, p1_y, p2_x, p2_y);
    } else if theta > 90.0 && theta < 180.0 {
        let theta_reverse = theta - 90.0;
        let theta_remaining = 90.0 - theta_reverse;

        // start
        if rho < 0.0 {
            p1_x = rho.abs() / theta_reverse.to_radians().sin();//deg2rad(theta_axis_size, theta_reverse).sin();
        } else {
            p1_x = rho.abs() * -1.0 / theta_reverse.to_radians().sin();//deg2rad(theta_axis_size, theta_reverse).sin();
        }

        p1_y = 0.0;

        // end
        p2_x = img_width as f64;

        if rho < 0.0 {
            p2_y = (img_width as f64 - p1_x.abs()) * theta_reverse.to_radians().sin() / (theta_remaining as f64).to_radians().sin();
        } else {
            p2_y = (img_width as f64 + p1_x.abs()) * theta_reverse.to_radians().sin() / (theta_remaining as f64).to_radians().sin();
        }
    }

    (p1_x, p1_y, p2_x, p2_y)
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    println!("args: {:?}", args);

    if args.len() < 4 {
        writeln!(io::stderr(), "Usage: hough-image input_path houghspace_img_path line_visualization_img_path rho_axis_scale_factor houghspace_filter_threshold").unwrap();
        process::exit(1);
    }

    // TODO:
    // argument parsing
    // improving edge detection
    // make computation of max. line length work for landscape and non-landscape
    // make more functional, split init()
    // allow configuration of theta_axis_size? for improved accuracy?
    // unit tests, especially for transform lines

    let input_img_path = args[0].to_string();
    let houghspace_img_path = args[1].to_string();
    let line_visualization_img_path = args[2].to_string();

    let rho_axis_scale_factor = u32::from_str(&args[3]).expect("ERROR 'rho_axis_scale_factor' argument not a number.");
    let houghspace_filter_threshold = u32::from_str(&args[4]).expect("ERROR 'houghspace_filter_threshold' argument not a number.");

    init(&input_img_path, &houghspace_img_path, &line_visualization_img_path, rho_axis_scale_factor, houghspace_filter_threshold);
}



// -- utility functions --

// Liang-Barsky function by Daniel White @ http://www.skytopia.com/project/articles/compsci/clipping.html
#[allow(unused_assignments)]
fn liang_barsky (
    edge_left: f64, edge_right: f64, edge_bottom: f64, edge_top: f64,   // Define the x/y clipping values for the border.
    x0src: f64, y0src: f64, x1src: f64, y1src: f64,                 // Define the start and end points of the line.
    x0clip: &mut f64, y0clip: &mut f64, x1clip: &mut f64, y1clip: &mut f64 // The output values, so declare these outside.
) -> bool {

    let mut t0: f64 = 0.0; let mut t1: f64 = 1.0;
    let xdelta = x1src - x0src;
    let ydelta = y1src - y0src;
    let mut p = 0.0f64;
    let mut q = 0.0f64;
    let mut r = 0.0f64;

    for edge in 0..4 {   // Traverse through left, right, bottom, top edges.
        if edge == 0 {  p = -xdelta;    q = -(edge_left - x0src);   }
        if edge == 1 {  p = xdelta;     q =  edge_right - x0src;    }
        if edge == 2 {  p = -ydelta;    q = -(edge_bottom - y0src); }
        if edge == 3 {  p = ydelta;     q =  edge_top - y0src;      }
        r = q/p;

        if p == 0.0 && q < 0.0 {
            // Don't draw line at all. (parallel line outside)
            return false;
        }

        if p < 0.0 {
            if r as f64 > t1 {
                // Don't draw line at all.
                return false;
            } else if r as f64 > t0 {
                // Line is clipped!
                t0 = r as f64;
            }
        } else if p > 0.0 {
            if (r as f64) < t0 {
                // Don't draw line at all.
                return false;
            }
            else if (r as f64) < t1 {
                // Line is clipped!
                t1 = r as f64;
            }
        }
    }

    *x0clip = x0src as f64 + t0 as f64 * xdelta as f64;
    *y0clip = y0src as f64 + t0 as f64 * ydelta as f64;
    *x1clip = x0src as f64 + t1 as f64 * xdelta as f64;
    *y1clip = y0src as f64 + t1 as f64 * ydelta as f64;

    return true;
}

// Based on http://stackoverflow.com/questions/34440429/draw-a-line-in-a-bitmap-possibly-with-piston
fn draw_line<T: GenericImage>(img: &mut T, x0: i32, y0: i32, x1: i32, y1: i32, pixel: T::Pixel) {
    // Create local variables for moving start point
    let mut x0 = x0;
    let mut y0 = y0;

    // Get absolute x/y offset
    let dx = if x0 > x1 { x0 - x1 } else { x1 - x0 };
    let dy = if y0 > y1 { y0 - y1 } else { y1 - y0 };

    // Get slopes
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };

    // Initialize error
    let mut err = if dx > dy { dx } else {-dy} / 2;
    let mut err2;

    loop {
        // Set pixel
        img.put_pixel(x0 as u32, y0 as u32, pixel);

        // Check end condition
        if x0 == x1 && y0 == y1 { break };

        // Store old error
        err2 = 2 * err;

        // Adjust error and start position
        if err2 > -dx { err -= dy; x0 += sx; }
        if err2 < dy { err += dx; y0 += sy; }
    }
}
