--[[ GUI for Model (MUI)
By Xiang Zhang @ New York University
Version 0.1, 03/07/2014
--]]

require("scroll")

local Mui = torch.class("Mui")

-- Constructor
-- config: (optional) config table of Mui
--    .width: (optional) width of scrollable window
--    .scale: (optional) scale of visualizing weights
--    .title: (optional) title of the scrollable window
--    .n: (optional) show only first n items for a module
function Mui:__init(config)
   local config = config or {}
   self.width = config.width or 800
   self.scale = config.scale or 1
   self.title = config.title or "Mui"
   self.n = config.n
   self.win = Scroll(self.width,self.title)
end

-- Visualize the weights of a sequential model
-- model: the sequential model
function Mui:drawSequential(model)
   self.win:clear()
   for i,m in ipairs(model.modules) do
      local t,w = self:readModule(m)
      self.win:drawText(tostring(i)..": "..t)
      for j,v in ipairs(w) do
	 self.win:drawImage(v,self.scale)
      end
   end
end

-- Get module texts and weights
-- m: module
function Mui:readModule(m)
   if torch.typename(m) == "nn.SpatialConvolution" then
      return self:readSpatialConvolution(m)
   elseif torch.typename(m) == "nn.Linear" then
      return self:readLinear(m)
   elseif torch.typename(m) == "nn.SpatialLPPooling" or
   torch.typename(m) == "nn.SpatialMaxPooling" then
      return self:readSpatialPooling(m)
   elseif torch.typename(m) == "nn.SpatialZeroPadding" then
      return self:readSpatialZeroPadding(m)
   elseif torch.typename(m) == "nn.Threshold" then
      return self:readThreshold(m)
   elseif torch.typename(m) == "nn.Dropout" then
      return self:readDropout(m)
   else
      return torch.typename(m),{}
   end
end

-- Read a spatial convolution module
function Mui:readSpatialConvolution(m)
   local t = torch.typename(m).." nInputPlane="..m.nInputPlane.." nOutputPlane="..m.nOutputPlane.." kW*kH="..m.kW.."*"..m.kH.." dW*dH="..m.dW.."*"..m.dH
   local w = {}

   -- Process the weights
   if m.weight:size(2) == 3 then
      for i = 1,m.weight:size(1) do
	 w[#w+1]=torch.Tensor(m.weight[i]:size()):copy(m.weight[i]):add(-m.weight[i]:mean()):div(6*m.weight[i]:std()):add(0.5)
	 if self.n and #w >= self.n then return t,w end
      end
   else
      for i = 1,m.weight:size(1) do
	 for j = 1,m.weight:size(2) do
	    w[#w+1] = torch.Tensor(m.weight[i][j]:size()):copy(m.weight[i][j]):add(-m.weight[i][j]:mean()):div(6*m.weight[i][j]:std()):add(0.5)
	    if self.n and #w >= self.n then return t,w end
	 end
      end
   end

   -- Process the bias
   w[#w+1] = torch.Tensor(1,m.nOutputPlane):copy(m.bias):add(-m.bias:mean()):div(6*m.bias:std()):add(0.5)

   return t, w
end

-- Read a spatial pooling module
function Mui:readSpatialPooling(m)
   local t = torch.typename(m).." kW*kH="..m.kW.."*"..m.kH.." dW*dH="..m.dW.."*"..m.dH
   return t,{}
end

-- Read a spatial zero padding module
function Mui:readSpatialZeroPadding(m)
   local t = torch.typename(m).." pad_l="..m.pad_l.." pad_r="..m.pad_r.." pad_t="..m.pad_t.." pad_b="..m.pad_b
   return t,{}
end

-- Read a threshold module
function Mui:readThreshold(m)
   local t = torch.typename(m).." threshold="..m.threshold.." val="..m.val
   return t,{}
end

-- Read a dropout module
function Mui:readDropout(m)
   local t = torch.typename(m).." p="..m.p
   return t,{}
end

-- Read a linear module
function Mui:readLinear(m)
   local t = torch.typename(m).." inputSize="..m.weight:size(2).." outputSize="..m.weight:size(1)
   return t,{}
end
