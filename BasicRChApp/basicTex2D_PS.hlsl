texture2D tex : register(t0);

SamplerState samp : register(s0);

struct PSInput {
  float4 pos : SV_POSITION;
  float3 localPos : POSITION0;
  float3 worldPos : POSITION1;
};

float4 main(PSInput psin) : SV_TARGET {
  float surfaceW = 3.56f;
  float surfaceH = 2.0f;
  float3 surfaceDims = float3(surfaceW, surfaceH, 0.0f);

  float2 texCoords = ((psin.localPos + (surfaceDims / 2.f)) / surfaceDims).xy;

  return tex.Sample(samp, texCoords).rgba;
}