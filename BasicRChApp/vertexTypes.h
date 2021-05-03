#pragma once

//***************** NEW *****************
//Added layout description to VertexPositionColor
//type from previous lab task.
//Added similar type for vertex containing position
//and normal vector.

#include <d3d11.h>
#include <DirectXMath.h>

namespace mini
{
	struct VertexPosition {
		DirectX::XMFLOAT3 position;

		static const D3D11_INPUT_ELEMENT_DESC Layout[1];
	};

	struct VertexPositionColor
	{
		DirectX::XMFLOAT3 position;
		DirectX::XMFLOAT3 color;

		static const D3D11_INPUT_ELEMENT_DESC Layout[2];
	};

	struct VertexPositionNormal
	{
		DirectX::XMFLOAT3 position;
		DirectX::XMFLOAT3 normal;

		static const D3D11_INPUT_ELEMENT_DESC Layout[2];
	};

	struct VertexPosNormTex2D {
		DirectX::XMFLOAT3 position;
		DirectX::XMFLOAT3 normal;
		DirectX::XMFLOAT2 textureCoords;

		static const D3D11_INPUT_ELEMENT_DESC Layout[3];
	};
}