cbuffer cbWorld : register(b0) { matrix worldMatrix; };

cbuffer cbView : register(b1) {
  matrix viewMatrix;
  matrix invViewMatrix;
};

cbuffer cbProj : register(b2) { matrix projMatrix; };

struct PSInput {
  float4 pos : SV_POSITION;
  float3 localPos : POSITION0;
  float3 worldPos : POSITION1;
};

PSInput main(float4 inpos : POSITION) {
  PSInput o;
  o.localPos = inpos.xyz;
  o.pos = mul(worldMatrix, inpos);
  o.worldPos = o.pos; // ptodo
  //o.pos = mul(viewMatrix, o.pos);
  o.pos = mul(projMatrix, o.pos);
	return o;
}