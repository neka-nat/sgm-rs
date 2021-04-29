extern crate image;
extern crate ndarray;

use image::{GrayImage, Luma};
use ndarray::{Array3, Array4};

struct ScanLine {
    drow: i32,
    dcol: i32,
    posdir: bool,
}

static PATH8: [ScanLine; 8] = [
    ScanLine {
        drow: 1,
        dcol: 1,
        posdir: true,
    },
    ScanLine {
        drow: 1,
        dcol: 0,
        posdir: true,
    },
    ScanLine {
        drow: 1,
        dcol: -1,
        posdir: true,
    },
    ScanLine {
        drow: 0,
        dcol: -1,
        posdir: false,
    },
    ScanLine {
        drow: -1,
        dcol: -1,
        posdir: false,
    },
    ScanLine {
        drow: -1,
        dcol: 0,
        posdir: false,
    },
    ScanLine {
        drow: -1,
        dcol: 1,
        posdir: false,
    },
    ScanLine {
        drow: 0,
        dcol: 1,
        posdir: true,
    },
];
static P1: u32 = 3;
static P2: u32 = 10;

fn calc_hamming_distance(val_l: &Luma<u8>, val_r: &Luma<u8>) -> u8 {
    let mut dist: u8 = 0;
    let Luma(d_l) = *val_l;
    let Luma(d_r) = *val_r;
    let mut d = d_l[0] ^ d_r[0];
    while d != 0 {
        d = d & (d - 1);
        dist += 1;
    }
    dist
}

fn census_transform(img: &GrayImage) -> GrayImage {
    let imgx = img.width();
    let imgy = img.height();
    let mut census = GrayImage::new(imgx, imgy);
    for x in 1..(imgx - 1) {
        for y in 1..(imgy - 1) {
            let center = img.get_pixel(x, y);
            let image::Luma(cdata) = *center;
            let mut val: u8 = 0;
            for dx in -1isize..2isize {
                for dy in -1isize..2isize {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let xdx = x as isize + dx;
                    let ydy = y as isize + dy;
                    let tmp = img.get_pixel(xdx as u32, ydy as u32);
                    let Luma(tdata) = *tmp;
                    val = (val + if tdata[0] < cdata[0] { 0 } else { 1 }) << 1;
                }
            }
            census.put_pixel(x, y, image::Luma([val]));
        }
    }
    census
}

fn calc_pixel_cost(census_l: &GrayImage, census_r: &GrayImage, d_range: usize) -> Array3<u8> {
    let imgx = census_l.width() as usize;
    let imgy = census_l.height() as usize;
    let mut ans = Array3::<u8>::zeros((imgy, imgx, d_range));
    for (x, y, val_l) in census_l.enumerate_pixels() {
        for d in 0..d_range {
            let du32 = d as u32;
            let mut val_r = &Luma([0u8]);
            if x as i32 - d as i32 >= 0 {
                val_r = census_r.get_pixel(x - du32, y);
            }
            ans[(y as usize, x as usize, d)] = calc_hamming_distance(val_l, val_r);
        }
    }
    ans
}

fn aggregate_cost(
    row: usize,
    col: usize,
    depth: usize,
    path: usize,
    cost_array: &Array3<u8>,
    agg_cost: &mut Array4<u32>,
) -> u32 {
    let mut val0: u32 = std::u32::MAX;
    let mut val1: u32 = std::u32::MAX;
    let mut val2: u32 = std::u32::MAX;
    let mut val3: u32 = std::u32::MAX;
    let mut min_prev_d: u32 = std::u32::MAX;
    let dcol = PATH8[path].dcol;
    let drow = PATH8[path].drow;
    let indiv_cost = cost_array[(row, col, depth)];
    let rows = cost_array.shape()[0] as i32;
    let cols = cost_array.shape()[1] as i32;
    let d_range = cost_array.shape()[2] as i32;
    if row as i32 - drow < 0
        || rows <= row as i32 - drow
        || col as i32 - dcol < 0
        || cols <= col as i32 - dcol
    {
        agg_cost[(path, row, col, depth)] = indiv_cost as u32;
        return agg_cost[(path, row, col, depth)];
    }
    for dd in 0..d_range {
        let prev = agg_cost[(
            path,
            (row as i32 - drow) as usize,
            (col as i32 - dcol) as usize,
            dd as usize,
        )];
        if prev < min_prev_d {
            min_prev_d = prev;
        }
        if depth == dd as usize {
            val0 = prev;
        } else if depth == (dd + 1) as usize {
            val1 = prev + P1;
        } else if depth == (dd - 1) as usize {
            val2 = prev + P1;
        } else {
            let tmp = prev + P2;
            if tmp < val3 {
                val3 = tmp;
            }
        }
    }
    let vals = vec![val0, val1, val2, val3];
    agg_cost[(path, row, col, depth)] =
        *vals.iter().min().unwrap() + indiv_cost as u32 - min_prev_d;
    agg_cost[(path, row, col, depth)]
}

fn aggregate_cost_for_each_scanline(
    cost_array: &Array3<u8>,
    agg_cost: &mut Array4<u32>,
    sum_cost: &mut Array3<u32>,
) {
    let rows = cost_array.shape()[0];
    let cols = cost_array.shape()[1];
    let d_range = cost_array.shape()[2];
    for row in 0..rows {
        for col in 0..cols {
            for d in 0..d_range {
                sum_cost[(row, col, d)] += PATH8
                    .iter()
                    .enumerate()
                    .filter_map(|(p, path)| {
                        if path.posdir {
                            Some(aggregate_cost(row, col, d, p, cost_array, agg_cost))
                        } else {
                            None
                        }
                    })
                    .sum::<u32>();
            }
        }
    }
    for row in (0..rows).rev() {
        for col in (0..cols).rev() {
            for d in 0..d_range {
                sum_cost[(row, col, d)] += PATH8
                    .iter()
                    .enumerate()
                    .filter_map(|(p, path)| {
                        if !path.posdir {
                            Some(aggregate_cost(row, col, d, p, cost_array, agg_cost))
                        } else {
                            None
                        }
                    })
                    .sum::<u32>();
            }
        }
    }
}

fn calc_disparity(sum_cost: &Array3<u32>, disp_img: &mut GrayImage) {
    let d_range = sum_cost.shape()[2];
    for (x, y, pixel) in disp_img.enumerate_pixels_mut() {
        let xusize = x as usize;
        let yusize = y as usize;
        let min_depth = (0..d_range)
            .min_by_key(|&d| sum_cost[(yusize, xusize, d)])
            .unwrap_or(0);
        *pixel = image::Luma([min_depth as u8]);
    }
}

pub fn compute_disp(left: &GrayImage, right: &GrayImage, d_range: usize) -> GrayImage {
    let left_g = image::imageops::blur(left, 1.0);
    let right_g = image::imageops::blur(right, 1.0);
    // left_g.save("debug0.png").unwrap();
    let census_l = census_transform(&left_g);
    let census_r = census_transform(&right_g);
    // census_l.save("debug1.png").unwrap();
    let cost_array = calc_pixel_cost(&census_l, &census_r, d_range);
    let imgx = census_l.width() as usize;
    let imgy = census_l.height() as usize;
    let mut disp_img = GrayImage::new(imgx as u32, imgy as u32);
    let mut agg_cost = Array4::<u32>::zeros((8, imgy, imgx, d_range));
    let mut sum_cost = Array3::<u32>::zeros((imgy, imgx, d_range));
    aggregate_cost_for_each_scanline(&cost_array, &mut agg_cost, &mut sum_cost);
    calc_disparity(&sum_cost, &mut disp_img);
    disp_img
}
