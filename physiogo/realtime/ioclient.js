class IOClient {

    socket = undefined
    
    constructor(address) {
        this.socket = io(address, { transports: ['websocket', 'polling', 'flashsocket'] })
    }

    on(msg, handler) {
        this.socket.on(msg, handler)
    }

    send(msg, data) {
        this.socket.emit(msg, {data: data})
    }

    disconnect() {
        this.socket.disconnect()
    }

}