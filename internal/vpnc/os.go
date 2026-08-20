package vpnc

import "runtime"

func osName() string {
	return runtime.GOOS
}
