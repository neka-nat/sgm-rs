extern crate image;
extern crate sgm_rs as sr;
use sr::compute_disp;

pub fn main() {
    let limg = image::open("examples/left.png").unwrap();
    let rimg = image::open("examples/right.png").unwrap();
    let limg_gray = limg.to_luma8();
    let rimg_gray = rimg.to_luma8();
    let d_range = 30;
    let mut disp_img = compute_disp(&limg_gray, &rimg_gray, d_range);
    for (_x, _y, pixel) in disp_img.enumerate_pixels_mut() {
        let image::Luma(tdata) = *pixel;
        *pixel = image::Luma([(tdata[0] as f32 / d_range as f32 * 255.0) as u8]);
    }

    disp_img.save("examples/disp.png").unwrap();
}
