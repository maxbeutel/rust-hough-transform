extern crate nalgebra as na;
extern crate image;
extern crate imageproc;

use std::io::Write;
use std::env;
use std::process;
use std::io;
use std::f64;
use std::fs::File;
use std::path::Path;
use std::str::FromStr;

use image::{ImageBuffer, Pixel, Rgba, GenericImage};
use imageproc::drawing::draw_line_segment_mut;

#[inline]
fn deg2rad(deg: u32, axis_size: u32) -> f64 {
    // compute radians based on the theta axis size, which can be greater than 180 deg
    let pi: f64 = std::f64::consts::PI;
    deg as f64 * (pi / axis_size as f64)
}

#[inline]
fn calculate_max_line_length(img_width: u32, img_height: u32) -> f64 {
    ((img_width as f64).hypot(img_height as f64)).ceil()
}

#[inline]
fn rgb_to_greyscale(r: u8, g: u8, b: u8) -> u8 {
    ((r as f64 + g as f64 + b as f64) / 3.0).round() as u8
}

#[inline]
fn is_edge(img: &image::DynamicImage, x: u32, y: u32) -> bool {
    let pixel = img.get_pixel(x, y);
    let greyscale_value = rgb_to_greyscale(pixel.channels()[0],
                                           pixel.channels()[1],
                                           pixel.channels()[2]);
    greyscale_value < 1
}

#[inline]
fn calculate_rho(theta: u32, theta_axis_size: u32, x: u32, y: u32) -> f64 {
    let sin = deg2rad(theta, theta_axis_size).sin();
    let cos = deg2rad(theta, theta_axis_size).cos();

    (x as f64) * cos + (y as f64) * sin
}

fn matrix_max<T: 'static>(matrix: &na::DMatrix<T>) -> Option<T>
    where T: std::cmp::Ord + na::core::Scalar
{
    matrix.data
        .iter()
        .max()
        .map(|m| *m)
}

fn dump_houghspace(accumulator: &na::DMatrix<u32>, houghspace_img_path: &str) {
    let max_accumulator_value = matrix_max(accumulator).unwrap_or(0);

    println!("# max accumulator value: {}", max_accumulator_value);

    let out_img_width = accumulator.nrows() as u32;
    let out_img_height = accumulator.ncols() as u32;

    let mut out = ImageBuffer::new(out_img_width, out_img_height);

    for y in 0..out_img_height {
        for x in 0..out_img_width {
            let n = na::min(((accumulator[(x as usize, y as usize)] as f64) * 255.0 /
                             (max_accumulator_value as f64))
                                .round() as u32,
                            255) as u8;
            let pixel = image::Rgb([n, n, n]);

            out[(x, out_img_height - y - 1)] = pixel;
        }
    }

    let ref mut fout = File::create(houghspace_img_path).unwrap();
    let _ = image::ImageRgb8(out).save(fout, image::PNG);
}

fn dump_line_visualization(mut img: &mut image::DynamicImage,
                           accumulator: &na::DMatrix<u32>,
                           theta_axis_scale_factor: u32,
                           houghspace_filter_threshold: u32,
                           line_visualization_img_path: &str) {
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

            let line_coordinates = line_from_rho_theta(theta as u32,
                                                       theta_axis_scale_factor,
                                                       rho as f64,
                                                       img_width,
                                                       img_height);

            lines.push((theta, line_coordinates));
        }
    }

    println!("# detected lines: {}", lines.len());

    let white = Rgba([255u8, 255u8, 255u8, 255u8]);

    for (_, line_coordinates) in lines {
        let clipped_line_coordinates =
            clip_line_liang_barsky((0, (img_width - 1) as i32, 0, (img_height - 1) as i32),
                                   line_coordinates)
                .expect("Line from rho/theta should be inside visible area of image.");

        // @TODO check if this makes sense
        draw_line_segment_mut(img,
                              // be sure to not overflow height
                              (clipped_line_coordinates.0 as f32,
                               (img_height as f32) - 1.0 - clipped_line_coordinates.1 as f32),
                              (clipped_line_coordinates.2 as f32,
                               (img_height as f32) - 1.0 - clipped_line_coordinates.3 as f32),
                              white);
    }

    let mut buffer = File::create(line_visualization_img_path).unwrap();
    let _ = img.save(&mut buffer, image::PNG);
}

fn hough_transform(img: &image::DynamicImage,
                   theta_axis_scale_factor: u32,
                   rho_axis_scale_factor: u32)
                   -> na::DMatrix<u32> {
    let (img_width, img_height) = img.dimensions();

    let max_line_length = calculate_max_line_length(img_width, img_height);

    let theta_axis_size = theta_axis_scale_factor * 180;
    let rho_axis_size = (max_line_length as u32) * rho_axis_scale_factor;

    let mut accumulator: na::DMatrix<u32> =
        na::DMatrix::from_element(theta_axis_size as usize, rho_axis_size as usize, 0);

    for y in 0..img_height {
        for x in 0..img_width {
            if is_edge(&img, x, y) {
                for theta in 0..theta_axis_size {
                    let y_inverted = img_height - y - 1;

                    let rho = calculate_rho(theta, theta_axis_size, x, y_inverted);
                    let rho_scaled = scale_rho(rho, rho_axis_size, max_line_length);

                    accumulator[(theta as usize, rho_scaled as usize)] += 1;
                }
            }
        }
    }

    accumulator
}



fn scale_rho(rho: f64, rho_axis_size: u32, max_line_length: f64) -> u32 {
    let rho_axis_half = (rho_axis_size as f64 / 2.0).round();
    ((rho * rho_axis_half / max_line_length).round() + rho_axis_half as f64) as u32
}

fn line_from_rho_theta(theta: u32,
                       theta_axis_scale_factor: u32,
                       rho: f64,
                       img_width: u32,
                       img_height: u32)
                       -> (i32, i32, i32, i32) {
    let mut p1_x = 0.0 as f64;
    let mut p1_y = 0.0 as f64;

    let mut p2_x = 0.0 as f64;
    let mut p2_y = 0.0 as f64;

    // here we scale theta back to "base 180", if theta scale factor was > 1
    let theta = (theta as f64 / theta_axis_scale_factor as f64).round();
    let theta_axis_size = 180;

    let alpha = theta % 90.0;
    let beta = 90.0 - alpha;

    // special cases - line is parallel to x/y axis
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
    // otherwise use law of sines to get lines
    } else if theta > 0.0 && theta < 90.0 {
        // start
        p1_x = 0.0;
        p1_y = rho.abs() / deg2rad(theta as u32, theta_axis_size).sin();

        // end
        p2_x = rho.abs() / deg2rad(beta as u32, theta_axis_size).sin();
        p2_y = 0.0;
    } else if theta > 90.0 && theta < 180.0 {
        // start
        if rho < 0.0 {
            p1_x = rho.abs() / deg2rad(alpha as u32, theta_axis_size).sin();
        } else {
            p1_x = rho.abs() * -1.0 / deg2rad(alpha as u32, theta_axis_size).sin();
        }

        p1_y = 0.0;

        // end
        p2_x = img_width as f64;

        if rho < 0.0 {
            p2_y = (img_width as f64 - p1_x.abs()) * deg2rad(alpha as u32, theta_axis_size).sin() / deg2rad(beta as u32, theta_axis_size).sin();
        } else {
            p2_y = (img_width as f64 + p1_x.abs()) * deg2rad(alpha as u32, theta_axis_size).sin() / deg2rad(beta as u32, theta_axis_size).sin();
        }
    }

    (p1_x.round() as i32, p1_y.round() as i32, p2_x.round() as i32, p2_y.round() as i32)
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();

    if args.len() < 4 {
        writeln!(io::stderr(),
                 "Usage: hough-image input_path houghspace_img_path line_visualization_img_path \
                  theta_axis_scale_factor rho_axis_scale_factor houghspace_filter_threshold")
            .unwrap();
        process::exit(1);
    }

    let input_img_path = args[0].to_string();
    let houghspace_img_path = args[1].to_string();
    let line_visualization_img_path = args[2].to_string();

    let theta_axis_scale_factor = u32::from_str(&args[3])
        .expect("ERROR 'theta_axis_scale_factor' argument not a number.");
    let rho_axis_scale_factor = u32::from_str(&args[4])
        .expect("ERROR 'rho_axis_scale_factor' argument not a number.");
    let houghspace_filter_threshold = u32::from_str(&args[5])
        .expect("ERROR 'houghspace_filter_threshold' argument not a number.");

    let mut img = image::open(&Path::new(&input_img_path)).expect("ERROR: input file not found.");
    // let mut accumulator: na::DMatrix<u32> = na::DMatrix::from_element(0, 0, 0);
    let accumulator = hough_transform(&mut img, theta_axis_scale_factor, rho_axis_scale_factor);

    dump_houghspace(&accumulator, &houghspace_img_path);
    dump_line_visualization(&mut img,
                            &accumulator,
                            theta_axis_scale_factor,
                            houghspace_filter_threshold,
                            &line_visualization_img_path);
}

// -- utility functions --

// Liang-Barsky function by Daniel White @ http://www.skytopia.com/project/articles/compsci/clipping.html
#[allow(unused_assignments)]
fn clip_line_liang_barsky(clipping_area: (i32, i32, i32, i32), /* Define the x/y clipping values for the border. */
                          line_coordinates: (i32, i32, i32, i32))
                          -> Option<(i32, i32, i32, i32)> {
    let (edge_left, edge_right, edge_bottom, edge_top) = clipping_area;
    let (x0src, y0src, x1src, y1src) = line_coordinates;

    let mut t0: f64 = 0.0;
    let mut t1: f64 = 1.0;

    let xdelta = (x1src as f64) - (x0src as f64);
    let ydelta = (y1src as f64) - (y0src as f64);

    let mut p = 0.0f64;
    let mut q = 0.0f64;
    let mut r = 0.0f64;

    for edge in 0..4 {
        // Traverse through left, right, bottom, top edges.
        if edge == 0 {
            p = -xdelta;
            q = -((edge_left as f64) - (x0src as f64));
        }
        if edge == 1 {
            p = xdelta;
            q = (edge_right as f64) - (x0src as f64);
        }
        if edge == 2 {
            p = -ydelta;
            q = -((edge_bottom as f64) - (y0src as f64));
        }
        if edge == 3 {
            p = ydelta;
            q = (edge_top as f64) - (y0src as f64);
        }
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
            } else if (r as f64) < t1 {
                // Line is clipped!
                t1 = r as f64;
            }
        }
    }

    let x0clip = (x0src as f64) + (t0 as f64) * (xdelta as f64);
    let y0clip = (y0src as f64) + (t0 as f64) * (ydelta as f64);
    let x1clip = (x0src as f64) + (t1 as f64) * (xdelta as f64);
    let y1clip = (y0src as f64) + (t1 as f64) * (ydelta as f64);

    Some((x0clip.round() as i32,
          y0clip.round() as i32,
          x1clip.round() as i32,
          y1clip.round() as i32))
}

#[cfg(test)]
mod test {
    use super::{clip_line_liang_barsky, line_from_rho_theta, calculate_rho};

    const CLIP_LINE_IMG_WIDTH: i32 = 500;
    const CLIP_LINE_IMG_HEIGHT: i32 = 300;

    #[test]
    fn test_clip_line_liang_barsky_1() {
        let p1 = (-200, -100);
        let p2 = (220, 400);

        let clipped = clip_line_liang_barsky((0, CLIP_LINE_IMG_WIDTH, 0, CLIP_LINE_IMG_HEIGHT),
                                             (p1.0, p1.1, p2.0, p2.1))
            .unwrap();

        assert_eq!((0, 138, 136, 300), clipped);
    }

    #[test]
    fn test_clip_line_liang_barsky_2() {
        let p1 = (300, -200);
        let p2 = (0, 390);

        let clipped = clip_line_liang_barsky((0, CLIP_LINE_IMG_WIDTH, 0, CLIP_LINE_IMG_HEIGHT),
                                             (p1.0, p1.1, p2.0, p2.1))
            .unwrap();

        assert_eq!((198, 0, 46, 300), clipped);
    }

    #[test]
    fn test_clip_line_liang_barsky_3() {
        let p1 = (400, 400);
        let p2 = (400, -150);

        let clipped = clip_line_liang_barsky((0, CLIP_LINE_IMG_WIDTH, 0, CLIP_LINE_IMG_HEIGHT),
                                             (p1.0, p1.1, p2.0, p2.1))
            .unwrap();

        assert_eq!((400, 300, 400, 0), clipped);

    }

    #[test]
    fn test_clip_line_liang_barsky_4() {
        let p1 = (200, 100);
        let p2 = (250, 190);

        let clipped = clip_line_liang_barsky((0, CLIP_LINE_IMG_WIDTH, 0, CLIP_LINE_IMG_HEIGHT),
                                             (p1.0, p1.1, p2.0, p2.1))
            .unwrap();

        assert_eq!((200, 100, 250, 190), clipped);
    }

    #[test]
    fn test_clip_line_liang_barsky_outside_of_clipping_window() {
        let p1 = (-200, -100);
        let p2 = (-250, -190);

        let clipped = clip_line_liang_barsky((0, CLIP_LINE_IMG_WIDTH, 0, CLIP_LINE_IMG_HEIGHT),
                                             (p1.0, p1.1, p2.0, p2.1));

        assert_eq!(None, clipped);
    }

    #[test]
    fn test_calculate_rho() {
        let theta_axis_size = 180;

        assert_eq!(50.0, calculate_rho(0, theta_axis_size, 50, 40).round());
        assert_eq!(64.0, calculate_rho(45, theta_axis_size, 50, 40).round());
        assert_eq!(40.0, calculate_rho(90, theta_axis_size, 50, 40).round());
        assert_eq!(-50.0, calculate_rho(180, theta_axis_size, 50, 40).round());
        assert_eq!(-40.0, calculate_rho(270, theta_axis_size, 50, 40).round());
        assert_eq!(-7.0, calculate_rho(135, theta_axis_size, 50, 40).round());
    }

    #[test]
    fn test_line_from_rho_theta_special_cases() {
        let img_width = 100;
        let img_height = 90;

        let x = 50;
        let y = 40;

        let theta_axis_size = 180;
        let theta_axis_scale_factor = 1;

        let line_coordinates = line_from_rho_theta(0,
                                                   theta_axis_scale_factor,
                                                   calculate_rho(0, theta_axis_size, x, y),
                                                   img_width,
                                                   img_height);

        assert_eq!((50, 90, 50, 0), line_coordinates);

        let line_coordinates = line_from_rho_theta(30,
                                                   theta_axis_scale_factor,
                                                   calculate_rho(30, theta_axis_size, x, y),
                                                   img_width,
                                                   img_height);

        assert_eq!((0, 127, 73, 0), line_coordinates);

        let line_coordinates = line_from_rho_theta(45,
                                                   theta_axis_scale_factor,
                                                   calculate_rho(45, theta_axis_size, x, y),
                                                   img_width,
                                                   img_height);

        assert_eq!((0, 90, 90, 0), line_coordinates);

        let line_coordinates = line_from_rho_theta(90,
                                                   theta_axis_scale_factor,
                                                   calculate_rho(90, theta_axis_size, x, y),
                                                   img_width,
                                                   img_height);

        assert_eq!((0, 40, 100, 40), line_coordinates);

        let line_coordinates = line_from_rho_theta(120,
                                                   theta_axis_scale_factor,
                                                   calculate_rho(120, theta_axis_size, x, y),
                                                   img_width,
                                                   img_height);

        assert_eq!((-19, 0, 100, 69), line_coordinates);

        let line_coordinates = line_from_rho_theta(135,
                                                   theta_axis_scale_factor,
                                                   calculate_rho(135, theta_axis_size, x, y),
                                                   img_width,
                                                   img_height);

        assert_eq!((10, 0, 100, 90), line_coordinates);

        let line_coordinates = line_from_rho_theta(180,
                                                   theta_axis_scale_factor,
                                                   calculate_rho(180, theta_axis_size, x, y),
                                                   img_width,
                                                   img_height);

        assert_eq!((50, 90, 50, 0), line_coordinates);

        // increase theta axis by two: then theta equals to 180 is the same as if theta is 90 for scale factor of 1
        let theta_axis_size = 360;
        let theta_axis_scale_factor = 2;

        let line_coordinates = line_from_rho_theta(180,
                                                   theta_axis_scale_factor,
                                                   calculate_rho(180, theta_axis_size, x, y),
                                                   img_width,
                                                   img_height);

        assert_eq!((0, 40, 100, 40), line_coordinates);
    }
}
