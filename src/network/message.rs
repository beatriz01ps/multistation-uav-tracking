use serde::Deserialize;

// é um atributo.
// É uma instrução/metainformação aplicada ao elemento que vem logo depois.
#[derive(Deserialize)]
pub struct ObservationMessage {
    pub message_id: String,
}