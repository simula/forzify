use crate::data::models::{Annotation, Attribute};
use opencv::core::Scalar;
use rand::RngCore;
use std::collections::HashMap;

/// Filters annotations based on team and category. Also handles if we ignore person_class or teams.
pub fn filter_annotations(
    image_id: &String,
    annotations: Vec<Annotation>,
    categories: &HashMap<String, u32>,
    ignore_person_class: bool,
    ignore_teams: bool,
) -> Vec<Annotation> {
    let default_player_id = 1;
    let cat_goalkeeper = categories.get("goalkeeper");
    let cat_referee = categories.get("referee");
    let cat_player = categories.get("player");

    annotations
        .into_iter()
        .filter_map(|annotation| {
            if annotation.image_id != *image_id {
                return None;
            }

            // Destructure for readability
            let Annotation {
                category_id,
                attributes,
                ..
            } = annotation;

            // If ignoring teams, just assign a random placeholder team id
            let new_team = if ignore_teams {
                Some(rand::rng().next_u32().to_string())
            } else {
                attributes.clone().unwrap_or_default().team
            };

            // If ignoring person_class, map "goalkeeper"/"referee" => "player"
            let new_category_id = if ignore_person_class {
                match category_id {
                    id if Some(&id) == cat_goalkeeper => *cat_player.unwrap_or(&default_player_id),
                    id if Some(&id) == cat_referee => *cat_player.unwrap_or(&default_player_id),
                    id => id,
                }
            } else {
                category_id
            };

            Some(Annotation {
                category_id: new_category_id,
                attributes: Some(Attribute {
                    role: attributes.as_ref().and_then(|a| a.role.clone()),
                    jersey: attributes.as_ref().and_then(|a| a.jersey.clone()),
                    team: new_team,
                }),
                ..annotation
            })
        })
        .collect()
}

pub fn get_annotation_color(annotation: &Annotation, categories: &HashMap<String, u32>) -> Scalar {
    let team_id = &annotation.attributes.as_ref().unwrap().team;

    if annotation.category_id == categories["ball"] {
        return Scalar::new(237.0, 237.0, 237.0, 255.0); // Gray for ball
    }

    // Determine color based on team_id
    match team_id.as_deref() {
        Some("left") => Scalar::new(0.0, 0.0, 255.0, 255.0), // Red for team A
        Some("right") => Scalar::new(255.0, 0.0, 0.0, 255.0), // Blue for team B
        _ => Scalar::new(0.0, 255.0, 0.0, 255.0),            // Default: Green
    }
}

/// Finds the closest annotation to the base annotation
pub fn annotation_comparator(
    base_annotation: Annotation,
    other_annotations: Vec<Annotation>,
    use_2d: bool,
) -> Option<Annotation> {
    let mut closest_annotation = None;
    let mut closest_distance = f64::MAX;

    for annotation in other_annotations {
        let distance =
            calculate_annotation_distance(base_annotation.clone(), annotation.clone(), use_2d)?;
        if distance < closest_distance {
            closest_distance = distance;
            closest_annotation = Some(annotation);
        }
    }

    closest_annotation
}

/// Determines if annotation center is within range
pub fn is_within_range(
    base_annotation: Annotation,
    other_annotation: Annotation,
    range: f64,
    use_2d: bool,
) -> Option<bool> {
    let distance = calculate_annotation_distance(base_annotation, other_annotation, use_2d)?;
    Some(distance < range)
}

/// Euclidean distance between two annotations
pub fn calculate_annotation_distance(
    annotation_1: Annotation,
    annotation_2: Annotation,
    use_2d: bool,
) -> Option<f64> {
    let coords_1 = calculate_bbox_pitch_coordinates(annotation_1, use_2d)?;
    let coords_2 = calculate_bbox_pitch_coordinates(annotation_2, use_2d)?;

    Some(((coords_2.0 - coords_1.0).powi(2) + (coords_2.1 - coords_1.1).powi(2)).sqrt())
}

/// Calculate the center of the BboxPitch
/// If `use_2d` is true, get the 2D center, if false get the bottom-center of the bounding box
pub fn calculate_bbox_pitch_coordinates(
    annotation: Annotation,
    use_2d: bool,
) -> Option<(f64, f64)> {
    let (x_center, y_center) = if use_2d {
        let bbox = annotation.bbox_pitch?;
        // Calculate the geometric center
        let x_center = (bbox.x_bottom_left + bbox.x_bottom_right) / 2.0;
        let y_center = (bbox.y_bottom_left + bbox.y_bottom_right) / 2.0;

        (x_center, y_center)
    } else {
        let bbox = annotation.bbox_image?;
        // Uses the bottom of the box, to use the feet of the player when not using 2d
        let x_center = bbox.x + (bbox.w / 2.0);
        let y_bottom = bbox.y + bbox.h;
        (x_center, y_bottom)
    };

    Some((x_center, y_center))
}

pub fn compute_average_player_bbox_height(
    annotations: &[Annotation],
    category_map: &HashMap<String, u32>,
) -> f64 {
    let player_cat_id = category_map.get("player").copied().unwrap_or(1);
    let mut total_height = 0.0;
    let mut count = 0;

    for ann in annotations {
        if ann.category_id == player_cat_id {
            if let Some(bi) = &ann.bbox_image {
                total_height += bi.h;
                count += 1;
            }
        }
    }
    if count > 0 {
        total_height / count as f64
    } else {
        1.0 // fallback if no players
    }
}

#[cfg(test)]
mod tests {
    use crate::data::models::BboxPitch;

    use super::*;

    fn create_annotation(x_bl: f64, y_bl: f64, x_br: f64, y_br: f64) -> Annotation {
        Annotation {
            bbox_pitch: Some(BboxPitch {
                x_bottom_left: x_bl,
                y_bottom_left: y_bl,
                x_bottom_right: x_br,
                y_bottom_right: y_br,
                x_bottom_middle: (x_bl + x_br) / 2.0,
                y_bottom_middle: (y_bl + y_br) / 2.0,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn test_calculate_bbox_pitch_center() {
        let annotation = create_annotation(0.0, 0.0, 2.0, 2.0);
        let center = calculate_bbox_pitch_coordinates(annotation, true).unwrap();
        assert_eq!(center, (1.0, 1.0));
    }

    #[test]
    fn test_calculate_annotation_distance() {
        let annotation_1 = create_annotation(0.0, 0.0, 2.0, 2.0); // center = (1, 1)
        let annotation_2 = create_annotation(3.0, 3.0, 5.0, 5.0); // center = (4, 4)
        let distance = calculate_annotation_distance(annotation_1, annotation_2, true).unwrap();
        assert!((distance - 4.242).abs() < 0.001); // sqrt((4-1)^2 + (4-1)^2) = 4.242
    }

    #[test]
    fn test_is_within_range() {
        let annotation_1 = create_annotation(0.0, 0.0, 2.0, 2.0); // center = (1, 1)
        let annotation_2 = create_annotation(1.0, 1.0, 3.0, 3.0); // center = (2, 2)
        let result = is_within_range(annotation_1, annotation_2, 2.0, true).unwrap();
        assert!(result); // sqrt((2-1)^2 + (2-1)^2) = sqrt(2) = 1.414 < 2.0
    }

    #[test]
    fn test_annotation_comparator() {
        let base_annotation = create_annotation(0.0, 0.0, 2.0, 2.0); // center = (1, 1)
        let other_annotations = vec![
            create_annotation(3.0, 3.0, 5.0, 5.0), // center = (4, 4)
            create_annotation(1.0, 1.0, 3.0, 3.0), // center = (2, 2)
            create_annotation(6.0, 6.0, 8.0, 8.0), // center = (7, 7)
        ];
        let closest_annotation =
            annotation_comparator(base_annotation, other_annotations, true).unwrap();
        let expected_annotation = create_annotation(1.0, 1.0, 3.0, 3.0);
        assert_eq!(
            closest_annotation.bbox_pitch.unwrap().x_bottom_left,
            expected_annotation.bbox_pitch.unwrap().x_bottom_left
        );
    }
}
