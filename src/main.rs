mod config;
mod network;

use std::io;

// fn main() {
//     println!("Hello, world!");
// }

fn main() -> io::Result<()> {

    let receiver_address = format!(
        "{}:{}",
        config::RECEIVER_IP,
        config::RECEIVER_PORT
    );

    network::receiver::start(&receiver_address)

    // aqui não preciso colocar o 
    //"network::receiver::start(&address)?;
    // Ok(())"
    // porque ele retorna direto o retorno de start, que já tem todo esse processo com "?" e "Ok(())"
}
