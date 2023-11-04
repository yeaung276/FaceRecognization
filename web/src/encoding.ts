import '@tensorflow/tfjs-core';
import { GraphModel, loadGraphModel } from "@tensorflow/tfjs-converter";
import { GRAPH_MODEL_PATH } from "./env";
import { Tensor } from '@tensorflow/tfjs';

export const WIDTH = 112;
export const HEIGHT = 112;

export default class Encoder{
    model: GraphModel | null = null

    constructor(){}

    async prepare(){
        this.model = await loadGraphModel(GRAPH_MODEL_PATH)
    }

    encode(face: Tensor){
        if(this.model === null){
            throw Error('Need to call prepare first to be able to encode faces.')
        }
        const b_tensor = face.expandDims(0).div(125.5).sub(1);
        return this.model.predict(b_tensor)
    }
}