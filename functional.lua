-- this file attempts to provide a purely functional set of bindings
-- all functions in this file retain absolutely no state.
-- There shouldn't be any reference to "self" in this file.

local cudnn = require 'cudnn.env'
local ffi = require 'ffi'
local errcheck = cudnn.errcheck
cudnn.functional = {}
error('cudnn.functional is obsolete, should not be used!')
