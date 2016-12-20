// Based on https://rosettacode.org/wiki/Hough_transform#Java

extern crate nalgebra as na;
extern crate image;

use std::io::Write;
use std::env;
use std::process;
use std::io;
use std::f64;
use std::fs::File;
use std::path::Path;
use std::str::FromStr;

use image::{ImageBuffer, Pixel};

#[inline]
fn deg2rad(deg: u32, axis_size: u32) -> f64 {
    // this is IMPORTANT:
    // compute radians based on the theta axis size, which can be larger than 180 deg!
    let pi: f64 = std::f64::consts::PI;
    deg as f64 * (pi / axis_size as f64)
}

fn calculate_max_line_length(img_width: u32, img_height: u32) -> f64 {
    ((img_width as f64).hypot(img_height as f64)).ceil()
}

fn rgb_to_greyscale(r: u8, g: u8, b: u8) -> u8 {
    ((r as f64 + g as f64 + b as f64) / 3.0).round() as u8
}

fn is_edge(img: &image::RgbImage, x: u32, y: u32) -> bool {
    let pixel = img.get_pixel(x, y);
    let i = rgb_to_greyscale(pixel.channels()[0], pixel.channels()[1], pixel.channels()[2]);
    i < 1
}

fn dump_houghspace(accumulator: &na::DMatrix<u32>, houghspace_img_path: &str) {
    let accu_clone = accumulator.clone().into_vector();
    let max_accumulator_value = *accu_clone.iter().max().unwrap();

    println!("max accu value: {}", max_accumulator_value);

    let out_img_width = accumulator.nrows() as u32;
    let out_img_height = accumulator.ncols() as u32;

    let mut out = ImageBuffer::new(out_img_width, out_img_height);

    for y in 0..out_img_height {
        for x in 0..out_img_width {
            let n = na::min(((accumulator[(x as usize, y as usize)] as f64) * 255.0 / (max_accumulator_value as f64)).round() as u32, 255) as u8;
            let pixel = image::Rgb([n, n, n]);

            out[(x, out_img_height - y - 1)] = pixel;
        }
    }

    let ref mut fout = File::create(&Path::new(houghspace_img_path)).unwrap();
    let _ = image::ImageRgb8(out).save(fout, image::PNG);
}

fn dump_line_visualization(
    mut img: &mut image::RgbImage,
    accumulator: &na::DMatrix<u32>,
    houghspace_filter_threshold: u32,
    line_visualization_img_path: &str
) {
    let (img_width, img_height) = img.dimensions();

    let theta_axis_size = accumulator.nrows();
    let rho_axis_size = accumulator.ncols();
    let rho_axis_half = ((rho_axis_size as f64) / 2.0).round();
    let max_line_length = calculate_max_line_length(img_width, img_height);

    let mut lines = vec![];

    for theta in 0..theta_axis_size {
        for rho_scaled in 0..rho_axis_size {
            let val = accumulator[(theta as usize, rho_scaled as usize)];

            if val < houghspace_filter_threshold {
                continue;
            }

            let rho = (rho_scaled as f64 - rho_axis_half) * max_line_length / rho_axis_half;

            let line_coordinates = line_from_rho_theta(
                theta as u32,
                theta_axis_size as u32,
                rho as f64,
                img_width,
                img_height
            );

            lines.push(line_coordinates);
        }
    }

    println!("detected lines: {}", lines.len());

    for line_coordinates in lines {
        let clipped_line_coordinates = clip_line_liang_barsky(
            (0, (img_width - 1) as i32, 0, (img_height - 1) as i32),
            line_coordinates
        ).expect("Line from rho/theta should be inside visible area of image.");

        let img_line_coordinates = (
            clipped_line_coordinates.0,
            (img_height as i32) - 1 - clipped_line_coordinates.1, // don't overflow height
            clipped_line_coordinates.2,
            (img_height as i32) - 1 - clipped_line_coordinates.3 // don't overflow height
        );

        draw_line(&mut img, img_line_coordinates);
    }

    let _ = img.save(&Path::new(line_visualization_img_path));
}

fn hough_transform(img: &image::RgbImage, rho_axis_scale_factor: u32) -> na::DMatrix<u32> {
    let (img_width, img_height) = img.dimensions();

    let max_line_length = calculate_max_line_length(img_width, img_height);

    let theta_axis_size = 1 * 180;

    // making rho axis size larger increases accuracy a lot (compare 8 * maxlen vs 2 * maxlen)
    // (preventing that different angles generate the same rho)
    let rho_axis_size = (max_line_length as u32) * rho_axis_scale_factor;
    let rho_axis_half = ((rho_axis_size as f64) / 2.0).round();

    let mut accumulator: na::DMatrix<u32> = na::DMatrix::new_zeros(theta_axis_size as usize, rho_axis_size as usize);

    for y in 0..img_height {
        for x in 0..img_width {
            let y_inverted = img_height - y - 1;

            if is_edge(&img, x, y) {
                for theta in 0..theta_axis_size {
                    let rho = calculate_rho(theta, theta_axis_size, x, y_inverted);
                    let rho_scaled = ((rho * rho_axis_half as f64 / max_line_length as f64).round() + rho_axis_half as f64) as u32;

                    accumulator[(theta as usize, rho_scaled as usize)] += 1;
                }
            }
        }
    }

    accumulator
}

fn calculate_rho(theta: u32, theta_axis_size: u32, x: u32, y: u32) -> f64 {
    let sin = deg2rad(theta, theta_axis_size).sin();
    let cos = deg2rad(theta, theta_axis_size).cos();

    (x as f64) * cos + (y as f64) * sin
}

#[test]
fn test_calculate_rho() {
    let theta_axis_size = 180;

    assert_eq!(50.0, calculate_rho(0.0, theta_axis_size, 50, 40).round());
    assert_eq!(64.0, calculate_rho(45.0, theta_axis_size, 50, 40).round());
    assert_eq!(40.0, calculate_rho(90.0, theta_axis_size, 50, 40).round());
    assert_eq!(-50.0, calculate_rho(180.0, theta_axis_size, 50, 40).round());
    assert_eq!(-40.0, calculate_rho(270.0, theta_axis_size, 50, 40).round());
    assert_eq!(-7.0, calculate_rho(135.0, theta_axis_size, 50, 40).round());

    println!("{}", deg2rad(90.0, 2*180));

    assert_eq!(40.0, calculate_rho(90.0, theta_axis_size, 50, 40).round());
}

fn line_from_rho_theta(
    theta: u32,
    theta_axis_size: u32,
    rho: f64,
    img_width: u32,
    img_height: u32
) -> (i32, i32, i32, i32) {
    let mut p1_x = 0.0 as f64;
    let mut p1_y = 0.0 as f64;

    let mut p2_x = 0.0 as f64;
    let mut p2_y = 0.0 as f64;

    // @FIXME this must be changed if we allow a different theta_axis_size
    let theta = if theta > 180 { theta as f64 - 180.0 } else { theta as f64 };

    let theta_reverse = theta % 90.0;
    let theta_remaining = 90.0 - theta_reverse;

    // special cases
    if theta == 0.0 || theta == 180.0 {
        p1_x = rho.abs();
        p1_y = img_height as f64;

        p2_x = rho.abs();
        p2_y = 0.0;
    } else if theta == 90.0 {
        p1_x = 0.0;
        p1_y = rho.abs();

        p2_x = img_width as f64;
        p2_y = rho.abs();
    } else if theta > 0.0 && theta < 90.0  {
        // start
        p1_x = 0.0;
        p1_y = rho.abs() / deg2rad(theta as u32, theta_axis_size).sin();

        // end
        p2_x = rho.abs() / deg2rad(theta_remaining as u32, theta_axis_size).sin();
        p2_y = 0.0;
    } else if theta > 90.0 && theta < 180.0 {
        // start
        if rho < 0.0 {
            p1_x = rho.abs() / deg2rad(theta_reverse as u32, theta_axis_size).sin();
        } else {
            p1_x = rho.abs() * -1.0 / deg2rad(theta_reverse as u32, theta_axis_size).sin();
        }

        p1_y = 0.0;

        // end
        p2_x = img_width as f64;

        if rho < 0.0 {
            p2_y = (img_width as f64 - p1_x.abs()) * deg2rad(theta_reverse as u32, theta_axis_size).sin() / deg2rad(theta_remaining as u32, theta_axis_size).sin();
        } else {
            p2_y = (img_width as f64 + p1_x.abs()) * deg2rad(theta_reverse as u32, theta_axis_size).sin() / deg2rad(theta_remaining as u32, theta_axis_size).sin();
        }
    }

    (p1_x.round() as i32, p1_y.round() as i32, p2_x.round() as i32, p2_y.round() as i32)
}

#[test]
fn test_line_from_rho_theta_special_cases() {
    let img_width = 100;
    let img_height = 90;

    let x = 50;
    let y = 40;

    let theta_axis_size = 180;

    let line_coordinates = line_from_rho_theta(0, theta_axis_size, calculate_rho(0.0, theta_axis_size, x, y), img_width, img_height);
    assert_eq!((50, 90, 50, 0), line_coordinates);

    let line_coordinates = line_from_rho_theta(30, theta_axis_size, calculate_rho(30.0, theta_axis_size, x, y), img_width, img_height);
    assert_eq!((0, 127, 73, 0), line_coordinates);

    let line_coordinates = line_from_rho_theta(45, theta_axis_size, calculate_rho(45.0, theta_axis_size, x, y), img_width, img_height);
    assert_eq!((0, 90, 90, 0), line_coordinates);

    let line_coordinates = line_from_rho_theta(90, theta_axis_size, calculate_rho(90.0, theta_axis_size, x, y), img_width, img_height);
    assert_eq!((0, 40, 100, 40), line_coordinates);

    let line_coordinates = line_from_rho_theta(120, theta_axis_size, calculate_rho(120.0, theta_axis_size, x, y), img_width, img_height);
    assert_eq!((-19, 0, 100, 69), line_coordinates);

    let line_coordinates = line_from_rho_theta(135, theta_axis_size, calculate_rho(135.0, theta_axis_size, x, y), img_width, img_height);
    assert_eq!((10, 0, 100, 90), line_coordinates);

    let line_coordinates = line_from_rho_theta(180, theta_axis_size, calculate_rho(180.0, theta_axis_size, x, y), img_width, img_height);
    assert_eq!((50, 90, 50, 0), line_coordinates);

    let line_coordinates = line_from_rho_theta(210, theta_axis_size, calculate_rho(210.0, theta_axis_size, x, y), img_width, img_height);
    assert_eq!((0, 127, 73, 0), line_coordinates);

    let line_coordinates = line_from_rho_theta(225, theta_axis_size, calculate_rho(225.0, theta_axis_size, x, y), img_width, img_height);
    assert_eq!((0, 90, 90, 0), line_coordinates);

    let line_coordinates = line_from_rho_theta(270, theta_axis_size, calculate_rho(270.0, theta_axis_size, x, y), img_width, img_height);
    assert_eq!((0, 40, 100, 40), line_coordinates);

    let line_coordinates = line_from_rho_theta(300, theta_axis_size, calculate_rho(300.0, theta_axis_size, x, y), img_width, img_height);
    assert_eq!((19, 0, 100, 47), line_coordinates);

    let line_coordinates = line_from_rho_theta(315, theta_axis_size, calculate_rho(315.0, theta_axis_size, x, y), img_width, img_height);
    assert_eq!((-10, 0, 100, 110), line_coordinates);

    let line_coordinates = line_from_rho_theta(360, theta_axis_size, calculate_rho(360.0, theta_axis_size, x, y), img_width, img_height);
    assert_eq!((50, 90, 50, 0), line_coordinates);
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    println!("args: {:?}", args);

    if args.len() < 4 {
        writeln!(io::stderr(), "Usage: hough-image input_path houghspace_img_path line_visualization_img_path rho_axis_scale_factor houghspace_filter_threshold").unwrap();
        process::exit(1);
    }

    // TODO:
    // [ ] improving edge detection
    // [ ] allow configuration of theta_axis_size? for improved accuracy? (need to fix line_from... function)
    // [X] argument parsing
    // [X] make computation of max. line length work for landscape and non-landscape
    // [X] make more functional, split init()
    // [X] more unit tests
    // [X] fix int overflow in line_from... function
    // [X] use f64 everywhere, absolutely everywhere!

    let input_img_path = args[0].to_string();
    let houghspace_img_path = args[1].to_string();
    let line_visualization_img_path = args[2].to_string();

    let rho_axis_scale_factor = u32::from_str(&args[3]).expect("ERROR 'rho_axis_scale_factor' argument not a number.");
    let houghspace_filter_threshold = u32::from_str(&args[4]).expect("ERROR 'houghspace_filter_threshold' argument not a number.");

    let mut img = image::open(&Path::new(&input_img_path)).expect("ERROR: input file not found.").to_rgb();
    let accumulator = hough_transform(&mut img, rho_axis_scale_factor);

    dump_houghspace(&accumulator, &houghspace_img_path);
    dump_line_visualization(&mut img, &accumulator, houghspace_filter_threshold, &line_visualization_img_path);
}

// -- utility functions --

// Liang-Barsky function by Daniel White @ http://www.skytopia.com/project/articles/compsci/clipping.html
#[allow(unused_assignments)]
fn clip_line_liang_barsky(
    clipping_area: (i32, i32, i32, i32), // Define the x/y clipping values for the border.
    line_coordinates: (i32, i32, i32, i32)
) -> Option<(i32, i32, i32, i32)> {
    let (edge_left, edge_right, edge_bottom, edge_top) = clipping_area;
    let (x0src, y0src, x1src, y1src) = line_coordinates;

    let mut t0: f64 = 0.0;
    let mut t1: f64 = 1.0;

    let xdelta = (x1src as f64) - (x0src as f64);
    let ydelta = (y1src as f64) - (y0src as f64);

    let mut p = 0.0f64;
    let mut q = 0.0f64;
    let mut r = 0.0f64;

    for edge in 0..4 {   // Traverse through left, right, bottom, top edges.
        if edge == 0 {  p = -xdelta;    q = -((edge_left as f64) - (x0src as f64));   }
        if edge == 1 {  p = xdelta;     q =  (edge_right as f64) - (x0src as f64);    }
        if edge == 2 {  p = -ydelta;    q = -((edge_bottom as f64) - (y0src as f64)); }
        if edge == 3 {  p = ydelta;     q =  (edge_top as f64) - (y0src as f64);      }
        r = q / p;

        if p == 0.0 && q < 0.0 {
            // Don't draw line at all. (parallel line outside)
            return None;
        }

        if p < 0.0 {
            if r as f64 > t1 {
                // Don't draw line at all.
                return None;
            } else if r as f64 > t0 {
                // Line is clipped!
                t0 = r as f64;
            }
        } else if p > 0.0 {
            if (r as f64) < t0 {
                // Don't draw line at all.
                return None;
            }
            else if (r as f64) < t1 {
                // Line is clipped!
                t1 = r as f64;
            }
        }
    }

    let x0clip = (x0src as f64) + (t0 as f64) * (xdelta as f64);
    let y0clip = (y0src as f64) + (t0 as f64) * (ydelta as f64);
    let x1clip = (x0src as f64) + (t1 as f64) * (xdelta as f64);
    let y1clip = (y0src as f64) + (t1 as f64) * (ydelta as f64);

    Some((x0clip.round() as i32, y0clip.round() as i32, x1clip.round() as i32, y1clip.round() as i32))
}

#[test]
fn test_clip_line_liang_barsky() {
    let img_width = 500;
    let img_height = 300;

    // Testcase A
    let p1 = (-200, -100);
    let p2 = (220, 400);

    let clipped = clip_line_liang_barsky(
        (0, img_width, 0, img_height),
        (p1.0, p1.1, p2.0, p2.1)
    ).unwrap();

    assert_eq!((0, 138, 136, 300), clipped);

    // Testcase B
    let p1 = (300, -200);
    let p2 = (0, 390);

    let clipped = clip_line_liang_barsky(
        (0, img_width, 0, img_height),
        (p1.0, p1.1, p2.0, p2.1)
    ).unwrap();

    assert_eq!((198, 0, 46, 300), clipped);

    // Testcase C
    let p1 = (400, 400);
    let p2 = (400, -150);

    let clipped = clip_line_liang_barsky(
        (0, img_width, 0, img_height),
        (p1.0, p1.1, p2.0, p2.1)
    ).unwrap();

    assert_eq!((400, 300, 400, 0), clipped);

    // Testcase D
    let p1 = (200, 100);
    let p2 = (250, 190);

    let clipped = clip_line_liang_barsky(
        (0, img_width, 0, img_height),
        (p1.0, p1.1, p2.0, p2.1)
    ).unwrap();

    assert_eq!((200, 100, 250, 190), clipped);

    // Testcase E - outside of clipping window
    let p1 = (-200, -100);
    let p2 = (-250, -190);

    let clipped = clip_line_liang_barsky(
        (0, img_width, 0, img_height),
        (p1.0, p1.1, p2.0, p2.1)
    );

    assert_eq!(None, clipped);
}

// Based on http://stackoverflow.com/questions/34440429/draw-a-line-in-a-bitmap-possibly-with-piston
fn draw_line(img: &mut image::RgbImage, line_coordinates: (i32, i32, i32, i32)) {
    let (x0, y0, x1, y1) = line_coordinates;

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
        img.put_pixel(x0 as u32, y0 as u32, image::Rgb([255, 0, 0]));

        // Check end condition
        if x0 == x1 && y0 == y1 { break };

        // Store old error
        err2 = 2 * err;

        // Adjust error and start position
        if err2 > -dx { err -= dy; x0 += sx; }
        if err2 < dy { err += dx; y0 += sy; }
    }
}
