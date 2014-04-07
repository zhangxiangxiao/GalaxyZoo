--[[ Model file for Galaxy Zoo
By Xiang Zhang @ New York University
--]]

-- Prerequisite
require("nn")

-- Create the class
local Model = torch.class("Model")

-- Constructor
-- config: the configuration table
--    [1],[2],...: the layered specification of model
--    .p: dropout probability
function Model:__init(config)
   -- Create a sequential for self
   if config.file then
      self.sequential = torch.load(config.file)
   else
      self.sequential = Model:createSequential(config)
   end
   self.p = config.p or 0.5
   self.tensortype = torch.getdefaulttensortype()
end

-- Get the parameters of the model
function Model:getParameters()
   return self.sequential:getParameters()
end

-- Forward propagation
function Model:forward(input)
   self.output = self.sequential:forward(input)
   return self.output
end

-- Backward propagation
function Model:backward(input, gradOutput)
   self.gradInput = self.sequential:backward(input, gradOutput)
   return self.gradInput
end

-- Randomize the model to random parameters
function Model:randomize(sigma)
   local w,dw = self:getParameters()
   w:normal():mul(sigma or 1)
end

-- Enable Dropouts
function Model:enableDropouts()
   self.sequential = self:changeSequentialDropouts(self.sequential, self.p)
end

-- Disable Dropouts
function Model:disableDropouts()
   self.sequential = self:changeSequentialDropouts(self.sequential,0)
end

-- Switch to a different data mode
function Model:type(tensortype)
   if tensortype == "torch.CudaTensor" then
      self.sequential:cuda()
      self.tensortype = tensortype
   elseif tensortype ~= nil then
      self.sequential:type(tensortype)
      self.tensortype = tensortype
   end
   return self.tensortype
end

-- Switch to cuda
function Model:cuda()
   self:type("torch.CudaTensor")
end

-- Switch to double
function Model:double()
   self:type("torch.DoubleTensor")
end

-- Switch to float
function Model:float()
   self:type("torch.FloatTensor")
end

-- Change dropouts
function Model:changeSequentialDropouts(model,p)
   for i,m in ipairs(model.modules) do
      if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
	  m.p = p
      end
   end
   return model
end

-- Create a sequential model using configurations
function Model:createSequential(model)
   local new = nn.Sequential()
   for i,m in ipairs(model) do
      new:add(Model:createModule(m))
   end
   return new
end

-- Create a module using configurations
function Model:createModule(m)
   if m.module == "nn.Reshape" then
      return Model:createReshape(m)
   elseif m.module == "nn.Linear" then
      return Model:createLinear(m)
   elseif m.module == "nn.Threshold" then
      return Model:createThreshold(m)
   elseif m.module == "nn.SpatialConvolution" then
      return Model:createSpatialConvolution(m)
   elseif m.module == "nn.SpatialMaxPooling" then
      return Model:createSpatialMaxPooling(m)
   elseif m.module == "nn.SpatialZeroPadding" then
      return Model:createSpatialZeroPadding(m)
   elseif m.module == "nn.Dropout" then
      return Model:createDropout(m)
   else
      error("Unrecognized module for creation: "..tostring(m.module))
   end
end

-- Create a new reshape model
function Model:createReshape(m)
   return nn.Reshape(m.size)
end

-- Create a new linear model
function Model:createLinear(m)
   return nn.Linear(m.inputSize, m.outputSize)
end

-- Create a new threshold model
function Model:createThreshold(m)
   return nn.Threshold()
end

-- Create a new Spatial Convolution model
function Model:createSpatialConvolution(m)
   return nn.SpatialConvolution(m.nInputPlane, m.nOutputPlane, m.kW, m.kH, m.dW, m.dH)
end

-- Create a new spatial max pooling model
function Model:createSpatialMaxPooling(m)
   return nn.SpatialMaxPooling(m.kW, m.kH, m.dW, m.dH)
end

-- Create a new Spatial Zeo Padding module
function Model:createSpatialZeroPadding(m)
   return nn.SpatialZeroPadding(m.pad_l,m.pad_r,m.pad_t,m.pad_b)
end

-- Create a new dropout module
function Model:createDropout(m)
   return nn.Dropout(m.p)
end
