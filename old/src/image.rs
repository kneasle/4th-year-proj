/// The specification for a full image
#[derive(Debug)]
pub struct Image {
    pub layers: Vec<Layer>,
}

#[derive(Debug)]
pub struct Layer {
    pub effects: Vec<EffectInstance>,
}

#[derive(Debug)]
pub struct EffectInstance {}
