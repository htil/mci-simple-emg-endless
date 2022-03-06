class p2 extends Phaser.Scene {
    constructor()
    {
        super({key: "p2"});
    }

    preload() {
        this.load.image('background', 'assets/sky.png');
        this.load.image('gameover', 'assets/gameover.png');
    }

    create() {
        this.bg = this.add.image(400,300,'background');
        this.gameover = this.add.image(400,300,'gameover');
        this.input.on('pointerdown', function (pointer) {
            this.scene.start('p1');
        }, this);
    }

    update(delta) {

    }
}