use super::dribble_models::{Ball, Player};
use crate::config::Config;
use crate::data::models::Annotation;
use crate::utils::annotation_calculations::calculate_bbox_pitch_coordinates;
use std::collections::HashMap;

pub fn get_ball_model(
    category_map: &HashMap<String, u32>,
    annotations: &[Annotation],
    config: &Config,
) -> Option<Ball> {
    let ball_id: u32 = match category_map.get("ball") {
        Some(id) => *id,
        None => return None,
    };

    let balls: Vec<Ball> = annotations
        .iter()
        .filter_map(|a| {
            if a.category_id == ball_id {
                let (x, y) =
                    calculate_bbox_pitch_coordinates(a.clone(), config.dribbling_detection.use_2d)?;
                Some(Ball { x: x, y: y })
            } else {
                None
            }
        })
        .collect();

    if balls.is_empty() {
        return None;
    }

    Some(balls[0])
}

pub fn get_player_models(
    category_map: &HashMap<String, u32>,
    annotations: &[Annotation],
    config: &Config,
) -> Option<Vec<Player>> {
    let player_id: u32 = match category_map.get("player") {
        Some(id) => *id,
        None => return None,
    };
    let players: Vec<Player> = annotations
        .iter()
        .filter_map(|a| {
            if a.category_id == player_id {
                let (x, y) =
                    calculate_bbox_pitch_coordinates(a.clone(), config.dribbling_detection.use_2d)?;
                Some(Player {
                    id: a.track_id.unwrap_or(u32::MAX).clone(),
                    x: x,
                    y: y,
                    velocity: (0.0, 0.0),
                    within_inner_rad: false,
                })
            } else {
                None
            }
        })
        .collect();

    Some(players)
}
