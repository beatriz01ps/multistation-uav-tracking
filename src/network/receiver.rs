use std::io;
use std::net::UdpSocket;


// & é uma referência, end e tamanho
pub fn start(address: &str) -> io::Result<()> {
    let socket =  UdpSocket::bind(address)?; // o "?" significa que se retornar um erro, ele para tudo e retornao erro agora msm

    println!("UDP receiver listening on {address}");

    loop {
        let mut buffer = [0_u8; 65_507]; // tamnho máximo teórico do conteúdo de um pacote UDP IPv4

        // desestruturação da tupla.                            // retorna a qtd de bytes e o endereço do remetente
        let (bytes_received, sender_address) = socket.recv_from(&mut buffer)?;

        println!("Received {bytes_received} bytes from {sender_address}");

        let received_data = &buffer[..bytes_received];

        let message = String::from_utf8_lossy(received_data);

        print!("Message: {message}");
    
    }

    // nao tem mais o ok pq o loop é "infinito" (até o programa ser encerrado ou ocorrer um erro)
    // Ok(())
}