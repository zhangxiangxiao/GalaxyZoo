--[[ Configuration file
By Xiang Zhang @ New York University
--]]

-- Create the namespace
config = {}

-- Training dataset
config.train_data = {}
config.train_data.file = paths.concat(paths.cwd(),"data/train.t7b")
config.train_data.root = paths.concat(paths.cwd(),"data/images_training_rev1")
config.train_data.crop = {method = "random",width=127,height=127}
config.train_data.rotate = 2*math.pi
config.train_data.scale = 1.5
config.train_data.batch_size = 128

-- Validation dataset
config.val_data = {}
config.val_data.file = paths.concat(paths.cwd(),"data/val.t7b")
config.val_data.root = paths.concat(paths.cwd(),"data/images_training_rev1")
config.val_data.crop = {method = "center",width=127,height=127}
config.val_data.batch_size = 128

-- The model configuration
config.model = {}
-- 3 x 127 x 127
config.model[1] = {module = "nn.SpatialConvolution", nInputPlane = 3, nOutputPlane = 64, kW = 5, kH = 5}
config.model[2] = {module = "nn.Threshold"}
config.model[3] = {module = "nn.SpatialMaxPooling", kW = 3, kH = 3}
-- 32 x 41 x 41
config.model[4] = {module = "nn.SpatialConvolution", nInputPlane = 64, nOutputPlane = 64, kW = 5, kH = 5}
config.model[5] = {module = "nn.Threshold"}
config.model[6] = {module = "nn.SpatialZeroPadding", pad_l = 1, pad_r = 1, pad_t = 1, pad_b = 1}
config.model[7] = {module = "nn.SpatialMaxPooling", kW = 3, kH = 3}
-- 64 x 13 x 13
config.model[8] = {module = "nn.SpatialConvolution", nInputPlane = 64, nOutputPlane = 128, kW = 3, kH = 3}
config.model[9] = {module = "nn.Threshold"}
config.model[10] = {module = "nn.SpatialZeroPadding", pad_l = 1, pad_r = 1, pad_t = 1, pad_b = 1}
-- 128 x 13 x 13
config.model[11] = {module = "nn.SpatialConvolution", nInputPlane = 128, nOutputPlane = 256, kW = 3, kH = 3}
config.model[12] = {module = "nn.Threshold"}
-- 256 x 11 x 11
config.model[13] = {module = "nn.SpatialConvolution", nInputPlane = 256, nOutputPlane = 512, kW = 3, kH = 3}
config.model[14] = {module = "nn.Threshold"}
config.model[15] = {module = "nn.SpatialMaxPooling", kW = 3, kH = 3}
-- 512 x 3 x 3
config.model[16] = {module = "nn.SpatialConvolution", nInputPlane = 512, nOutputPlane = 1024, kW = 3, kH = 3}
config.model[17] = {module = "nn.Threshold"}
-- 1024 x 1 x 1
config.model[18] = {module = "nn.Reshape", size = 1024*1*1}
-- 1024
config.model[19] = {module = "nn.Linear", inputSize = 1024, outputSize = 2048}
config.model[20] = {module = "nn.Threshold"}
config.model[21] = {module = "nn.Dropout", p = 0.5}
-- 2048
config.model[22] = {module = "nn.Linear", inputSize = 2048, outputSize = 2048}
config.model[23] = {module = "nn.Threshold"}
config.model[24] = {module = "nn.Dropout", p = 0.5}
-- 2048
config.model[25] = {module = "nn.Linear", inputSize = 2048, outputSize = 37}

-- The loss configuration
config.loss = nn.MSECriterion

-- The training configuration
config.train = {}
local baseRate = 1.25e-2
config.train.rates = {[1] = baseRate}
config.train.normalize = true

-- The testing configuration
config.test = {}
config.test.normalize = true

-- The UI configuration
config.mui = {}
config.mui.width = 1200
config.mui.scale = 4
config.mui.n = 256

-- The main configuration
config.main = {}
config.main.type = "torch.CudaTensor"
config.main.epoches = 5000
config.main.eras = 100
config.main.randomize = 1e-2
config.main.save = paths.concat(paths.cwd(),"models")
config.main.collectgarbage = 100
config.main.details = true
config.main.logtime = 5
config.main.device = 1
config.main.dropout = true
config.main.debug = false
config.main.test = true
