use super::message::ObservationMessage;


//          nome : tipo da variável recebida (& é ref;e u8 é um inteiro de 8 bits unsigned)
//                           result<tipo retornado certo, tipo retornado erro>
pub fn parse(data: &[u8]) -> Result<ObservationMessage, serde_json::Error> {
    serde_json::from_slice(data)

    // mesma coisa que escrever de forma explícita:
    // serde_json::from_slice::<ObservationMessage>(data)
    // o compilador sabe que queremos retornar <ObservationMessage> por conta do RESULT
}