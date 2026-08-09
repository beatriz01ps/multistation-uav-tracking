use std::io;
use std::net::UdpSocket;
use super::message_parser;

// & é uma referência, end e tamanho
pub fn start(address: &str) -> io::Result<()> {
    let socket =  UdpSocket::bind(address)?; // o "?" significa que se retornar um erro, ele para tudo e retornao erro agora msm

    println!("UDP receiver listening on {address}");
     
    let mut buffer = [0_u8; 65_507]; // tamnho máximo teórico do conteúdo de um pacote UDP IPv4

    loop {
        // desestruturação da tupla.                            // retorna a qtd de bytes e o endereço do remetente
        let (bytes_received, sender_address) = socket.recv_from(&mut buffer)?;

        println!("Received {bytes_received} bytes from {sender_address}");

        let received_data = &buffer[..bytes_received];

        // let message = String::from_utf8_lossy(received_data);

        // print!("Message: {message}");

        let observation = match message_parser::parse(received_data) {
            Ok(observation)=> observation,
            Err(error)=> {
                eprintln!("Failed to parse message: {error}");
                continue;
            }
        };

        println!(
            "Received message {} from {}",
            observation.message_id, sender_address
        );
    
    }

    // nao tem mais o ok pq o loop é "infinito" (até o programa ser encerrado ou ocorrer um erro)
    // Ok(())
}