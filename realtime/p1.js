const PIPES_TO_RENDER = 4;

let state = "";

client.on('prediction', (args) => {
    state = args;
})

class p1 extends Phaser.Scene {
    constructor()
    {
        super({key: "p1"});
    }
    preload() {
        this.load.image('background', 'assets/sky.png');
        this.load.image('box', 'assets/box.png');
        this.load.image('pipe', 'assets/pipe.png');
        this.load.image('gameover', 'assets/gameover.png')
    }

    create() {
        this.bg = this.add.image(200,150,'background');
        this.box = this.physics.add.sprite(200,150,'box');
        this.box.setGravityY(500);
        this.createPipes();
        this.box.setCollideWorldBounds(true);
        this.box.body.onWorldBounds = true;
        //this.box.body.world.on('worldbounds', this.restartGame ,this);

        let squeezeString = new String(state["data"]);
        //console.log(squeezeString);
        //console.log(state["data"]);

        if(squeezeString=='Squeeze')
        {
            //console.log("Yes");
            this.box.setVelocityY(-400);
        }

        this.input.on('pointerdown', () => {
            this.box.setVelocityY(-400);
        });
    }



    createPipes() {
        this.pipes = this.physics.add.group()
        this.physics.add.overlap(this.box, this.pipes, this.restartGame, null, this);
        this.pipeHorizontalPosition = 400;
        this.pipeVerticalPosition = 100;

        this.pipeHorizontalDist = Phaser.Math.Between(400, 500);

        for(let i = 0; i < PIPES_TO_RENDER; i++) {
            const upperPipe = this.pipes.create(0, 0, 'pipe').setOrigin(0, 1);
            const lowerPipe = this.pipes.create(0, 0, 'pipe').setOrigin(0, 0);
            this.placePipe(upperPipe, lowerPipe);
        }
        this.pipes.setVelocityX(-200);
    }

    placePipe(upperPipe, lowerPipe) {
        const rightMostPipeX = this.getRightMostPipe();
        this.pipeHorizontalDist = Phaser.Math.Between(400, 500);

        upperPipe.x = rightMostPipeX + this.pipeHorizontalDist;
        upperPipe.y = Phaser.Math.Between(300, 350);
        //lowerPipe.x = upperPipe.x;
        //lowerPipe.y = upperPipe.y + 125;
    }

    recyclePipes() {
        const tempPipes = [];

        this.pipes.getChildren().forEach(pipe => {
            if(pipe.getBounds().right < 0) {
                tempPipes.push(pipe);
                if(tempPipes.length === 2) {
                    this.placePipe(tempPipes[0], tempPipes[1]);
                }
            }
        });
    }

    getRightMostPipe() {
        let rightMost = 0;
        this.pipes.getChildren().forEach(pipe => {
            rightMost = Math.max(pipe.x, rightMost);
        });
        return rightMost;
    }

    restartGame() {
        this.registry.destroy();
        this.events.off();
        this.scene.restart();
    }

    update(delta) {
        // this.deltaSum += delta;
        // if(this.deltaSum>=1000) {
        //     console.log(this.box.body.velocity.y);
        //     this.deltaSum = 0;
        // }
        // this.add.text(0, 0, this.score, { font: '"Press Start 2P"' });
        this.recyclePipes();
    }
}
