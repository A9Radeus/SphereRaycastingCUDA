struct PSInput {
  float4 pos : SV_POSITION;
  float4 norm : NORMAL;
  // float3 localPos : POSITION0;
  // float3 worldPos : POSITION1;
};

float4 main(PSInput psin) : SV_TARGET
{
	//return float4(1.0f, 0.5f, 0.5f, 1.0f);
	return psin.norm;
}