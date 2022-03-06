



var config = {
    type: Phaser.AUTO,
    width: 550,
    height: 350,
    physics: {
        default: 'arcade',
        arcade: {
            gravity: {y: 0}
        }
    },
    scene: [p1]
};

var game = new Phaser.Game(config);

