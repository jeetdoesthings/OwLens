import SwiftUI

/// Black launch / loading splash — Helvetica “OwLens” while camera initializes.
struct SplashView: View {
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            Text("OwLens")
                .font(.custom("Helvetica", size: 42))
                .fontWeight(.regular)
                .foregroundColor(.white)
                .tracking(2)

            VStack {
                Spacer()
                HStack(spacing: 8) {
                    Image("github_logo")
                        .resizable()
                        .renderingMode(.template)
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 16, height: 16)
                        .foregroundColor(.white.opacity(0.4))
                    Text("jeetdoesthings")
                        .font(.custom("Helvetica", size: 14))
                        .foregroundColor(.white.opacity(0.4))
                        .tracking(1)
                }
                .padding(.bottom, 40)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
    }
}

#Preview {
    SplashView()
}
